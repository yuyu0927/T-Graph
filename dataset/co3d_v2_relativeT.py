import gzip
import io
import json
import os.path as osp
import random
from typing import Dict, List, Optional, Sequence, Union

import lmdb
import numpy as np
import torch
from PIL import Image, ImageFile
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import Dataset
from torchvision import transforms

from utils.bbox import square_bbox
from utils.misc import get_permutations
from utils.normalize_cameras import first_camera_transform, normalize_cameras

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CO3D_DIR = "/gpfs/work2/0/prjs0801/co3d/dataset"
CO3D_ANNOTATION_DIR = "/gpfs/work2/0/prjs0801/co3d/annotations"

# ---------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------
TRAINING_CATEGORIES = [
    "apple", "backpack", "banana", "baseballbat", "baseballglove", "bench",
    "bicycle", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "cup", "donut", "hairdryer", "handbag", "hydrant",
    "keyboard", "laptop", "microwave", "motorcycle", "mouse", "orange",
    "parkingmeter", "pizza", "plant", "stopsign", "teddybear", "toaster",
    "toilet", "toybus", "toyplane", "toytrain", "toytruck", "tv", "umbrella",
    "vase", "wineglass",
]

TEST_CATEGORIES = [
    "ball", "book", "couch", "frisbee", "hotdog",
    "kite", "remote", "sandwich", "skateboard", "suitcase",
]

# ---------------------------------------------------------------------
# PIL safety
# ---------------------------------------------------------------------
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

AnnoItem = Dict[str, Union[str, List[float], List[int]]]


class Co3dDataset(Dataset):
    """
    CO3D sequence dataset (LMDB-backed).

    Output keys:
      - image: (V, 3, H, W)
      - crop_params: (V, 3)   [-ccx, -ccy, crop_width] in normalized coords
      - R, T: (V, 3, 3), (V, 3)  (normalized if normalize_cameras=True)
      - (optional) R_original, T_original
      - (optional) relative_rotation: (P, 3, 3)
      - (optional) relative_translation: (P, 3) (unit-normalized by max norm)
    """

    def __init__(
        self,
        category: Sequence[str] = ("all",),
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        debug: bool = False,
        random_aug: bool = True,
        jitter_scale: Sequence[float] = (1.1, 1.2),
        jitter_trans: Sequence[float] = (-0.07, 0.07),
        num_images: int = 2,
        img_size: int = 224,
        random_num_images: bool = True,  # (kept for compatibility, not used)
        eval_time: bool = False,
        normalize_cameras_flag: bool = False,
        first_camera_transform_flag: bool = False,
        first_camera_rotation_only: bool = False,
        mask_images: bool = False,
    ):
        if "all" in category:
            category = TRAINING_CATEGORIES
        self.categories = sorted(category)

        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got: {split}")
        self.split = split

        if eval_time:
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)

        self.debug = debug
        self.num_images = int(num_images)
        self.image_size = int(img_size)
        self.eval_time = eval_time

        self.normalize_cameras = bool(normalize_cameras_flag)
        self.first_camera_transform = bool(first_camera_transform_flag)
        self.first_camera_rotation_only = bool(first_camera_rotation_only)
        self.mask_images = bool(mask_images)

        # Aug params (freeze jitter during eval_time)
        if random_aug and not eval_time:
            self.jitter_scale = tuple(jitter_scale)
            self.jitter_trans = tuple(jitter_trans)
        else:
            self.jitter_scale = (1.15, 1.15)
            self.jitter_trans = (0.0, 0.0)

        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.seq_data: Dict[str, List[AnnoItem]] = {}
        self.seq_to_category: Dict[str, str] = {}
        self.low_quality_translations: List[str] = []

        self._load_annotations(split_name=self.split)
        self.sequence_list = list(self.seq_data.keys())

        # LMDB cache: open once per category (important for speed)
        self._lmdb_envs: Dict[str, lmdb.Environment] = {}

        if self.debug:
            print(f"[Co3dDataset] split={self.split}, categories={len(self.categories)}")
            print(f"[Co3dDataset] low-quality seq skipped: {len(self.low_quality_translations)}")
            print(f"[Co3dDataset] sequences used: {len(self.sequence_list)}")

    # -----------------------
    # Lifecycle
    # -----------------------
    def __del__(self):
        for env in getattr(self, "_lmdb_envs", {}).values():
            try:
                env.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.sequence_list)

    # -----------------------
    # IO helpers
    # -----------------------
    def _load_annotations(self, split_name: str) -> None:
        """
        Read and filter annotation files:
          - drop sequences with len < num_images
          - drop sequences with unreasonable translations
        """
        for c in self.categories:
            ann_path = osp.join(CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")
            with gzip.open(ann_path, "r") as fin:
                annotation = json.loads(fin.read())

            for seq_name, seq_items in annotation.items():
                if len(seq_items) < self.num_images:
                    continue

                filtered: List[AnnoItem] = []
                bad_seq = False
                for item in seq_items:
                    T = item["T"]
                    if (T[0] + T[1] + T[2]) > 1e5:
                        bad_seq = True
                        self.low_quality_translations.append(seq_name)
                        break

                    filtered.append(
                        {
                            "filepath": item["filepath"],
                            "bbox": item["bbox"],
                            "R": item["R"],
                            "T": item["T"],
                            "focal_length": item["focal_length"],
                            "principal_point": item["principal_point"],
                        }
                    )

                if not bad_seq:
                    self.seq_data[seq_name] = filtered
                    self.seq_to_category[seq_name] = c

    def _get_lmdb_env(self, category: str) -> lmdb.Environment:
        if category not in self._lmdb_envs:
            env_path = osp.join(CO3D_DIR, f"lmdb_{category}")
            self._lmdb_envs[category] = lmdb.Environment(
                env_path, create=False, readonly=True, lock=False
            )
        return self._lmdb_envs[category]

    @staticmethod
    def _read_pil_from_lmdb(txn: lmdb.Transaction, key: str, mode: str) -> Image.Image:
        buf = txn.get(key.encode())
        if buf is None:
            raise FileNotFoundError(f"LMDB missing key: {key}")
        return Image.open(io.BytesIO(buf)).convert(mode)

    # -----------------------
    # Geometry helpers
    # -----------------------
    def _jitter_bbox(self, bbox_xyxy: np.ndarray) -> np.ndarray:
        """
        bbox_xyxy: [x0,y0,x1,y1] -> jittered square bbox int
        """
        bbox = square_bbox(bbox_xyxy.astype(np.float32))
        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side
        extent = side / 2 * s

        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    @staticmethod
    def _crop_image(image: Image.Image, bbox_xyxy: np.ndarray, white_bg: bool) -> Image.Image:
        x0, y0, x1, y1 = bbox_xyxy.tolist()
        w, h = x1 - x0, y1 - y0

        if white_bg:
            canvas = Image.new("RGB", (w, h), (255, 255, 255))
            canvas.paste(image, (-x0, -y0))
            return canvas

        return transforms.functional.crop(image, top=y0, left=x0, height=h, width=w)

    @staticmethod
    def _compute_relative_rt(
        rotations: List[torch.Tensor],
        translations: List[torch.Tensor],
        permutations: List[tuple],
        eps: float = 1e-12,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute relative rotation and translation for each (i, j):
            R_ij = R_i^T R_j
            t_ij = t_j - R_ij t_i
        Then normalize t_ij by max ||t_ij|| across pairs (to stabilize scale).
        """
        device = rotations[0].device
        dtype = torch.float32

        P = len(permutations)
        rel_R = torch.zeros((P, 3, 3), device=device, dtype=dtype)
        rel_t = torch.zeros((P, 3), device=device, dtype=dtype)

        for k, (i, j) in enumerate(permutations):
            Ri = rotations[i].to(dtype)
            Rj = rotations[j].to(dtype)
            ti = translations[i].to(dtype)
            tj = translations[j].to(dtype)

            Rij = Ri.T @ Rj
            tij = tj - Rij @ ti

            rel_R[k] = Rij
            rel_t[k] = tij

        norms = torch.norm(rel_t, dim=1)
        max_norm = torch.clamp(norms.max(), min=eps)
        rel_t = rel_t / max_norm

        return {"relative_rotation": rel_R, "relative_translation": rel_t}

    # -----------------------
    # Public API
    # -----------------------
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq_name = self.sequence_list[index]
        meta = self.seq_data[seq_name]
        ids = np.random.choice(len(meta), self.num_images, replace=False)
        return self.get_data(sequence_name=seq_name, ids=ids)

    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
        ids: Sequence[int] = (0, 1),
        no_images: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if sequence_name is None:
            if index is None:
                raise ValueError("Either index or sequence_name must be provided.")
            sequence_name = self.sequence_list[index]

        meta = self.seq_data[sequence_name]
        category = self.seq_to_category[sequence_name]
        annos = [meta[i] for i in ids]

        # Fast path: only R/T
        if no_images:
            R = torch.stack([torch.tensor(a["R"]) for a in annos])
            T = torch.stack([torch.tensor(a["T"]) for a in annos])
            return {"R": R, "T": T}

        # 1) Read PIL images (+ masks) from LMDB
        env = self._get_lmdb_env(category)
        images_pil: List[Image.Image] = []
        rotations: List[torch.Tensor] = []
        translations: List[torch.Tensor] = []

        with env.begin(write=False) as txn:
            for a in annos:
                filepath = a["filepath"]
                image = self._read_pil_from_lmdb(txn, filepath, mode="RGB")

                if self.mask_images:
                    white = Image.new("RGB", image.size, (255, 255, 255))
                    mask_name = osp.basename(filepath.replace(".jpg", ".png"))
                    mask_key = osp.join(category, sequence_name, "masks", mask_name)
                    mask = self._read_pil_from_lmdb(txn, mask_key, mode="L")
                    if mask.size != image.size:
                        mask = mask.resize(image.size)

                    mask = Image.fromarray((np.array(mask) > 125).astype(np.uint8) * 255)
                    image = Image.composite(image, white, mask)

                images_pil.append(image)
                rotations.append(torch.tensor(a["R"]))
                translations.append(torch.tensor(a["T"]))

        # 2) Crop -> transform, and compute crop_params
        crop_params: List[torch.Tensor] = []
        images_tensor: List[torch.Tensor] = []

        for a, img in zip(annos, images_pil):
            h, w = img.height, img.width
            bbox = np.array(a["bbox"])
            bbox_j = self._jitter_bbox(bbox)

            img_crop = self._crop_image(img, bbox_j, white_bg=self.mask_images)
            img_t = self.transform(img_crop) if self.transform is not None else img_crop
            images_tensor.append(img_t)

            crop_center = (bbox_j[:2] + bbox_j[2:]) / 2
            cc = (2 * crop_center / min(h, w)) - 1
            crop_w = 2 * (bbox_j[2] - bbox_j[0]) / min(h, w)
            crop_params.append(torch.tensor([-cc[0], -cc[1], crop_w], dtype=torch.float32))

        batch: Dict[str, torch.Tensor] = {
            "model_id": sequence_name,
            "category": category,
            "n": torch.tensor(len(meta)),
            "ind": torch.tensor(ids),
            "crop_params": torch.stack(crop_params),
            "image": torch.stack(images_tensor) if self.transform is not None else images_tensor,
        }

        # 3) Cameras (normalized or raw) + relative pose pairs
        if self.normalize_cameras:
            cams = PerspectiveCameras(
                focal_length=[a["focal_length"] for a in annos],
                principal_point=[a["principal_point"] for a in annos],
                R=[a["R"] for a in annos],
                T=[a["T"] for a in annos],
            )

            norm_cams, *_ = normalize_cameras(cams)
            if norm_cams == -1:
                raise RuntimeError("normalize_cameras failed: camera scale was 0")

            if self.first_camera_transform or self.first_camera_rotation_only:
                norm_cams = first_camera_transform(
                    norm_cams, rotation_only=self.first_camera_rotation_only
                )

            batch["R"] = norm_cams.R
            batch["T"] = norm_cams.T
            batch["R_original"] = torch.stack([torch.tensor(a["R"]) for a in annos])
            batch["T_original"] = torch.stack([torch.tensor(a["T"]) for a in annos])

            perms = get_permutations(len(ids), eval_time=self.eval_time)
            rel = self._compute_relative_rt(rotations, translations, perms)
            batch.update(rel)
        else:
            batch["R"] = torch.stack(rotations)
            batch["T"] = torch.stack(translations)

        return batch
