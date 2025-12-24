import json
import os
import os.path as osp
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras
from tqdm.auto import tqdm

from models.util import get_model
from utils import compute_optimal_alignment, compute_optimal_translation_alignment
from .eval_rotation import ORDER_PATH
from .util import get_dataset


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get_device() -> str:
    return "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def _full_scene_scale(R: torch.Tensor, T: torch.Tensor, device: torch.device) -> float:
    """
    Scene scale: max distance from camera centers to centroid.
    R: (N,3,3), T: (N,3)
    """
    cams = FoVPerspectiveCameras(R=R.to(device), T=T.to(device), device=device)
    cc = cams.get_camera_center()  # (N,3)
    centroid = cc.mean(dim=0)
    scale = torch.linalg.norm(cc - centroid, dim=1).max().item()
    return float(scale)


def _get_n_consistent_cameras_from_rel(R_pred_rel: np.ndarray, num_frames: int) -> torch.Tensor:
    """
    Convert relative rotations (for permutations where i==0) to absolute rotations with R0 = I.
    In your original code: k loops over get_permutations(...), and if i==0, assign R_abs[j] = R_pred[k].
    Here we assume R_pred_rel is stacked in the same order as get_permutations(num_frames, eval_time=True).
    """
    from utils import get_permutations  # keep local to avoid circular imports if any

    R_abs = torch.zeros((num_frames, 3, 3), dtype=torch.float32)
    R_abs[0] = torch.eye(3, dtype=torch.float32)

    perms = get_permutations(num_frames, eval_time=True)
    assert len(R_pred_rel) == len(perms), "R_pred_rel length must match permutations length"

    for k, (i, j) in enumerate(perms):
        if i == 0:
            R_abs[j] = torch.from_numpy(R_pred_rel[k]).to(torch.float32)
    return R_abs


def _compute_error(
    mode: str,
    R_pred: torch.Tensor,
    T_pred: torch.Tensor,
    R_gt: torch.Tensor,
    T_gt: torch.Tensor,
    gt_scene_scale: float,
    device: torch.device,
) -> Tuple[List[float], torch.Tensor]:
    """
    mode:
      - "cc": camera-center error after optimal alignment
      - "t" : translation error after optimal translation alignment (given R_pred)
    Returns:
      norms: python list of per-frame normalized errors
      A_hat: aligned prediction (camera centers or translations depending on mode)
    """
    if mode == "cc":
        cams_gt = FoVPerspectiveCameras(R=R_gt.to(device), T=T_gt.to(device), device=device)
        cc_gt = cams_gt.get_camera_center()
        cams_pr = FoVPerspectiveCameras(R=R_pred.to(device), T=T_pred.to(device), device=device)
        cc_pr = cams_pr.get_camera_center()

        A_hat, _, _, _ = compute_optimal_alignment(cc_gt, cc_pr)
        denom = max(gt_scene_scale, 1e-12)
        norm = torch.linalg.norm(cc_gt - A_hat, dim=1) / denom
        return norm.detach().cpu().numpy().tolist(), A_hat

    if mode == "t":
        T_A_hat, _, _ = compute_optimal_translation_alignment(T_gt, T_pred, R_pred)
        denom = max(gt_scene_scale, 1e-12)
        norm = torch.linalg.norm(T_gt - T_A_hat, dim=1) / denom
        return norm.detach().cpu().numpy().tolist(), T_A_hat

    raise ValueError(f"Unknown mode '{mode}', expected 'cc' or 't'.")


# ---------------------------------------------------------------------
# Main evaluation entry
# ---------------------------------------------------------------------
def evaluate_category_translation(
    checkpoint_path: str,
    category: str,
    mode: str,
    num_frames: int,
    use_pbar: bool = False,
    force: bool = False,
    sample_num: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Translation evaluation (consistent with your original logic):
      1) predict T_pred using the model
      2) read R_pred_rel from rotation results (coordinate_ascent) JSON
      3) convert to absolute rotations R_pred_n (R0=I)
      4) compute error (mode: "cc" or "t"), normalized by GT scene scale
      5) save per-sequence results

    Returns:
      translation_errors: (N_total_frames,) numpy array
    """
    save_dir = osp.join(checkpoint_path, f"eval/{mode}-{num_frames:03d}-sample{sample_num}")
    os.makedirs(save_dir, exist_ok=True)
    out_path = osp.join(save_dir, f"{category}.json")

    if osp.exists(out_path) and not force:
        print(f"{out_path} already exists, skipping")
        data = _load_json(out_path)
        errs = []
        for d in data.values():
            errs.extend(d["errors"])
        return np.asarray(errs, dtype=np.float32)

    device_str = _get_device()
    device = torch.device(device_str)

    # Load model
    model, _ = get_model(model_dir=checkpoint_path, device=device_str)

    # Load dataset in annotation camera frame (same as your original)
    dataset = get_dataset(category=category, num_images=num_frames, eval_time=True)

    # Deterministic frame order (same as rotation eval)
    order_path = ORDER_PATH.format(sample_num=sample_num, category=category)
    order = _load_json(order_path)

    # Rotation predictions JSON (same category + same sample_num + same num_frames)
    rot_json_path = osp.join(
        checkpoint_path,
        f"eval/coordinate_ascent-{num_frames:03d}-sample{sample_num}",
        f"{category}.json",
    )
    if not osp.exists(rot_json_path):
        raise FileNotFoundError(
            "Rotation results not found. Expected:\n"
            f"  {rot_json_path}\n"
            "Run rotation evaluation (coordinate_ascent) first."
        )
    rotations_json = _load_json(rot_json_path)

    iterable = tqdm(dataset) if use_pbar else dataset

    all_errors: Dict[str, Any] = {}
    translation_errors: List[float] = []

    for metadata in iterable:
        seq_name = metadata["model_id"]
        key_frames = order[seq_name][:num_frames]

        # --- GT scene scale computed using *all* cameras of the sequence (annotation frame)
        all_cams = dataset.get_data(
            sequence_name=seq_name,
            ids=np.arange(0, int(metadata["n"])),
            no_images=True,
        )
        gt_scene_scale = _full_scene_scale(all_cams["R"], all_cams["T"], device=device)

        # --- Inputs for the selected key frames
        batch = dataset.get_data(sequence_name=seq_name, ids=key_frames)
        images = batch["image"].to(device).unsqueeze(0)            # (1,V,3,H,W)
        crop_params = batch["crop_params"].to(device).unsqueeze(0) # (1,V,3)
        R_gt = batch["R"].to(device)                               # (V,3,3)
        T_gt = batch["T"].to(device)                               # (V,3)

        # --- Predict translation
        with torch.no_grad():
            out = model(images=images, crop_params=crop_params)
            # Your original: _, _, T_pred, _ = out
            # Keep the same unpacking style but make it robust:
            if not isinstance(out, (list, tuple)) or len(out) < 3:
                raise RuntimeError("Model output is unexpected; expected a tuple/list with T_pred at index 2.")
            T_pred = out[2].to(device)  # expected shape: (V,3) or (1,V,3) depending on your model

        # Normalize T_pred shape to (V,3)
        if T_pred.ndim == 3 and T_pred.shape[0] == 1:
            T_pred = T_pred.squeeze(0)

        # --- Read predicted rotations for this sequence and convert to absolute rotations
        if seq_name not in rotations_json:
            raise KeyError(f"Sequence '{seq_name}' not found in rotation json: {rot_json_path}")

        R_pred_rel = np.asarray(rotations_json[seq_name]["R_pred_rel"], dtype=np.float32)
        R_pred_n = _get_n_consistent_cameras_from_rel(R_pred_rel, num_frames).to(device)  # (V,3,3)

        # --- Compute translation error
        norms, A_hat = _compute_error(
            mode=mode,
            R_pred=R_pred_n,
            T_pred=T_pred,
            R_gt=R_gt,
            T_gt=T_gt,
            gt_scene_scale=gt_scene_scale,
            device=device,
        )

        translation_errors.extend(norms)
        all_errors[seq_name] = {
            "R_pred": R_pred_n.detach().cpu().numpy().tolist(),
            "T_pred": T_pred.detach().cpu().numpy().tolist(),
            "errors": norms,
            "scale": gt_scene_scale,
            "A_hat": A_hat.detach().cpu().numpy().tolist(),
            "key_frames": key_frames,
        }

    _save_json(out_path, all_errors)

    errs = np.asarray(translation_errors, dtype=np.float32)
    print("Average translation error:", float(np.mean(errs)))
    print("Acc(<0.2):", float(np.mean(errs < 0.2)))
    return errs
