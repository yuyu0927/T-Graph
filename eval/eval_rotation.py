import json
import os
import os.path as osp
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from models.util import get_model
from utils import generate_random_rotations, get_permutations
from util import compute_angular_error_batch, get_dataset

# ---------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Cached rotations for deterministic evaluation. If missing, random rotations are used.
PROPOSALS_PATH = osp.join(_THIS_DIR, "../rotations.pt")

# Pre-computed frame order per sequence
ORDER_PATH = "data/co3d_v2_random_order_{sample_num}/{category}.json"

# Large-scale proposals for pairwise search
NUM_PAIRWISE_QUERIES = 500_000


# ---------------------------------------------------------------------
# Small utils
# ---------------------------------------------------------------------
def _get_device() -> str:
    return "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"


def _load_proposals(device: torch.device, num_queries: int) -> torch.Tensor:
    """
    Load cached proposal rotations if available, otherwise generate random ones.
    Returns: (Q, 3, 3) float32 on device
    """
    if osp.exists(PROPOSALS_PATH):
        props = torch.load(PROPOSALS_PATH, map_location="cpu")
        props = props.to(device=device, dtype=torch.float32)
        if props.shape[0] < num_queries:
            # If cache is smaller, fall back to random to match requested size
            props = generate_random_rotations(num_queries, device=str(device))
        else:
            props = props[:num_queries]
    else:
        props = generate_random_rotations(num_queries, device=str(device))
    return props


def _extract_features(model, num_frames: int, images: torch.Tensor, crop_params: torch.Tensor) -> torch.Tensor:
    """
    images: (1, V, 3, H, W)
    crop_params: (1, V, 3)
    return: (1, V, C, 1, 1)
    """
    crop_pe = model.positional_encoding(crop_params)
    feats = model.feature_extractor(images, crop_pe=crop_pe)  # (1,V,C,1,1) or (V,C,1,1)
    return feats.reshape((1, num_frames, model.full_feature_dim, 1, 1))


def _relative_rt_from_abs(
    R_abs: torch.Tensor,  # (V,3,3)
    T_abs: torch.Tensor,  # (V,3)
    num_frames: int,
    eval_time: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each (i,j) permutation:
      R_ij = R_i^T R_j
      t_ij = t_j - R_ij t_i
    Normalize t_ij by max norm across all pairs.
    """
    perms = get_permutations(num_frames, eval_time=eval_time)
    P = len(perms)
    device = R_abs.device
    rel_R = torch.empty((P, 3, 3), device=device, dtype=torch.float32)
    rel_t = torch.empty((P, 3), device=device, dtype=torch.float32)

    R_abs = R_abs.to(torch.float32)
    T_abs = T_abs.to(torch.float32)

    for k, (i, j) in enumerate(perms):
        Rij = R_abs[i].T @ R_abs[j]
        tij = T_abs[j] - Rij @ T_abs[i]
        rel_R[k] = Rij
        rel_t[k] = tij

    max_norm = torch.clamp(torch.norm(rel_t, dim=1).max(), min=eps)
    rel_t = rel_t / max_norm
    return rel_R, rel_t


# ---------------------------------------------------------------------
# Core inference utilities
# ---------------------------------------------------------------------
def initialize_graph(
    num_frames: int,
    model,
    images: torch.Tensor,
    crop_params: torch.Tensor,
    proposals: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute best relative rotation and probability for all ordered pairs (i,j).
    Returns:
      best_rotations: (V,V,3,3) numpy
      best_probs:     (V,V)     numpy
    """
    feats = _extract_features(model, num_frames, images, crop_params)
    best_rot = np.zeros((num_frames, num_frames, 3, 3), dtype=np.float32)
    best_p = np.zeros((num_frames, num_frames), dtype=np.float32)

    with torch.no_grad():
        for i in range(num_frames):
            for j in range(num_frames):
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                _, _, R_best, p_best = model.predict_probability(
                    feats[0, a], feats[0, b], queries=proposals
                )
                if i > j:
                    R_best = R_best.T
                best_rot[i, j] = R_best.detach().cpu().numpy()
                best_p[i, j] = float(p_best)

    return best_rot, best_p


def compute_mst(
    num_frames: int,
    best_probs: np.ndarray,      # (V,V)
    best_rotations: np.ndarray,  # (V,V,3,3)
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Prim-like MST on a directed complete graph, picking the max-prob edge each step.
    Returns:
      assigned_rotations: (V,3,3) absolute rotations (up to global gauge)
      edges: list of chosen directed edges
    """
    assigned = {0}
    R_abs = np.tile(np.eye(3, dtype=np.float32), (num_frames, 1, 1))
    edges: List[Tuple[int, int]] = []

    while len(assigned) < num_frames:
        best_i, best_j = -1, -1
        best_p = -1.0
        not_assigned = set(range(num_frames)) - assigned

        for i in assigned:
            for j in not_assigned:
                # consider i->j
                if best_probs[i, j] > best_p:
                    best_p = float(best_probs[i, j])
                    best_i, best_j = i, j
                # consider j->i
                if best_probs[j, i] > best_p:
                    best_p = float(best_probs[j, i])
                    best_i, best_j = j, i

        rot_ij = best_rotations[best_i, best_j]  # R_i^T R_j (directional)
        if best_i in assigned:
            assigned.add(best_j)
            R_abs[best_j] = R_abs[best_i] @ rot_ij
        else:
            assigned.add(best_i)
            R_abs[best_i] = R_abs[best_j] @ rot_ij.T

        edges.append((best_i, best_j))

    return R_abs, edges


def n_to_np_rotations(num_frames: int, R_abs: np.ndarray) -> np.ndarray:
    """
    Convert absolute rotations (V,3,3) to relative rotations stacked by permutations.
    Return: (P,3,3)
    """
    perms = get_permutations(num_frames, eval_time=True)
    rel = [R_abs[i].T @ R_abs[j] for (i, j) in perms]
    return np.stack(rel, axis=0).astype(np.float32)


# ---------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------
def evaluate_pairwise(model, images: torch.Tensor, crop_params: torch.Tensor) -> Tuple[np.ndarray, Any]:
    """
    Evaluate per pair independently by brute-force over proposals.
    Returns: R_pred_rel (P,3,3), aux=None
    """
    device = images.device
    num_frames = images.shape[1]
    perms = get_permutations(num_frames, eval_time=True)

    model.num_queries = NUM_PAIRWISE_QUERIES
    feats = _extract_features(model, num_frames, images, crop_params)
    proposals = _load_proposals(device, NUM_PAIRWISE_QUERIES)

    rots_pred: List[np.ndarray] = []
    with torch.no_grad():
        for i, j in perms:
            a, b = (i, j) if i < j else (j, i)
            _, _, R_best, _ = model.predict_probability(feats[0, a], feats[0, b], queries=proposals)
            if i > j:
                R_best = R_best.T
            rots_pred.append(R_best.detach().cpu().numpy())

    return np.stack(rots_pred, axis=0).astype(np.float32), None


def coordinate_ascent(
    num_frames: int,
    model,
    images: torch.Tensor,
    crop_params: torch.Tensor,
    initial_hypothesis: np.ndarray,     # (V,3,3)
    num_iterations: int = 50,
    num_queries: int = 250_000,
    use_pbar: bool = True,
    double_count_logits: bool = False,  # keep compatibility with your old "scores += logits" twice
) -> torch.Tensor:
    """
    Refine absolute rotations by coordinate ascent.
    Returns: hypothesis (V,3,3) torch tensor on device
    """
    device = images.device
    model.num_queries = num_queries

    hypothesis = torch.from_numpy(initial_hypothesis).to(device=device, dtype=torch.float32)
    feats = _extract_features(model, num_frames, images, crop_params)

    it = tqdm(range(num_iterations)) if use_pbar else range(num_iterations)
    for _ in it:
        k = int(np.random.choice(num_frames))

        proposals = generate_random_rotations(num_queries, device=str(device))
        proposals[0] = hypothesis[k]  # keep current as a candidate

        scores = torch.zeros((1, num_queries), device=device, dtype=torch.float32)

        with torch.no_grad():
            for i in range(num_frames):
                if i == k:
                    continue

                a, b = (i, k) if i < k else (k, i)
                R_rel = hypothesis[i].T @ proposals            # (Q,3,3) via broadcast
                R = R_rel if i < k else R_rel.transpose(1, 2) # align query direction

                _, logits, _, _ = model.predict_probability(
                    feature1=feats[0, a],
                    feature2=feats[0, b],
                    queries=R,
                    take_softmax=False,
                )
                scores += logits
                if double_count_logits:
                    scores += logits

        best_ind = int(scores.argmax())
        hypothesis[k] = proposals[best_ind]

    return hypothesis


def evaluate_coordinate_ascent(model, images: torch.Tensor, crop_params: torch.Tensor, use_pbar: bool = False):
    """
    Initialize with MST, then coordinate ascent refinement.
    Returns: R_pred_rel (P,3,3), aux=list(abs rotations)
    """
    device = images.device
    num_frames = images.shape[1]

    # Build complete pair graph once
    proposals = _load_proposals(device, NUM_PAIRWISE_QUERIES)
    model.num_queries = NUM_PAIRWISE_QUERIES
    best_rots, best_probs = initialize_graph(num_frames, model, images, crop_params, proposals)

    # MST initialization
    R_abs_init, _ = compute_mst(num_frames, best_probs, best_rots)

    # Coordinate ascent refinement
    R_abs = coordinate_ascent(
        num_frames=num_frames,
        model=model,
        images=images,
        crop_params=crop_params,
        initial_hypothesis=R_abs_init,
        use_pbar=use_pbar,
    ).detach().cpu().numpy()

    R_pred_rel = n_to_np_rotations(num_frames, R_abs)
    return R_pred_rel, R_abs.tolist()


def _get_eval_fn(mode: str) -> Callable:
    mapping = {
        "pairwise": evaluate_pairwise,
        "coordinate_ascent": evaluate_coordinate_ascent,
        # "mst": evaluate_mst,  # if you want to expose it later
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(mapping.keys())}")
    return mapping[mode]


# ---------------------------------------------------------------------
# Main evaluation entry
# ---------------------------------------------------------------------
def evaluate_category_rotation(
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
    Evaluate a category and save per-sequence predictions/GT/errors into json.

    Returns:
      angular_errors: (N_total_pairs,) numpy array
    """
    save_dir = osp.join(checkpoint_path, f"eval/{mode}-{num_frames:03d}-sample{sample_num}")
    os.makedirs(save_dir, exist_ok=True)
    out_path = osp.join(save_dir, f"{category}.json")

    # Fast return if cached
    if osp.exists(out_path) and not force:
        print(f"{out_path} exists, skipping")
        with open(out_path, "r") as f:
            data = json.load(f)
        angular_errors = []
        for d in data.values():
            angular_errors.extend(d["angular_errors"])
        return np.asarray(angular_errors, dtype=np.float32)

    # Load model
    device_str = _get_device()
    device = torch.device(device_str)
    model, _ = get_model(model_dir=checkpoint_path, device=device_str)

    # Load dataset
    dataset = get_dataset(
        category=category,
        num_images=num_frames,
        eval_time=True,
        normalize_cameras=True,
    )

    # Load deterministic frame order
    order_path = ORDER_PATH.format(sample_num=sample_num, category=category)
    with open(order_path, "r") as f:
        order = json.load(f)

    eval_fn = _get_eval_fn(mode)

    iterable = tqdm(dataset) if use_pbar else dataset

    all_errors: Dict[str, Any] = {}
    angular_errors: List[float] = []

    for metadata in iterable:
        seq_name = metadata["model_id"]
        key_frames = order[seq_name][:num_frames]

        batch = dataset.get_data(sequence_name=seq_name, ids=key_frames)

        # Ground truth relative rotations
        # batch["relative_rotation"]: (P,3,3) already
        R_gt_rel = batch["relative_rotation"].detach().cpu().numpy().astype(np.float32)

        # Inputs
        images = batch["image"].to(device).unsqueeze(0)         # (1,V,3,H,W)
        crop_params = batch["crop_params"].to(device).unsqueeze(0)  # (1,V,3)

        # Predict
        R_pred_rel, aux = eval_fn(model, images, crop_params)

        # Errors
        errors = compute_angular_error_batch(R_pred_rel, R_gt_rel).astype(np.float32)

        angular_errors.extend(errors.tolist())
        all_errors[seq_name] = {
            "R_pred_rel": R_pred_rel.tolist(),
            "R_gt_rel": R_gt_rel.tolist(),
            "angular_errors": errors.tolist(),
            "key_frames": key_frames,
            "aux": aux,  # optional: abs rotations or None
        }

    # Save
    with open(out_path, "w") as f:
        json.dump(all_errors, f)

    errs = np.asarray(angular_errors, dtype=np.float32)
    print("Acc(<15°):", float(np.mean(errs < 15)))
    print("Acc(<30°):", float(np.mean(errs < 30)))
    return errs
