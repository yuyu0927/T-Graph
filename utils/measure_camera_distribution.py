import torch
from torch import Tensor
from torch.nn import functional as F


def compute_optical_axis_directions(cameras) -> Tensor:
    """
    Compute normalized optical-axis direction vectors for all cameras.
    Assumes principal_point is in NDC when from_ndc=True.
    Returns:
        r: (N, 3) normalized axis directions
    """
    centers = cameras.get_camera_center()              # (N, 3)
    principal_points = cameras.principal_point         # (N, 2)
    N = centers.shape[0]

    one = torch.ones((N, 1), device=centers.device, dtype=centers.dtype)
    optical_axis = torch.cat((principal_points, one), dim=-1)  # (N, 3)

    # Unproject to world; take diagonal without Python loop
    pp = cameras.unproject_points(optical_axis, from_ndc=True, world_coordinates=True)  # (N, N, 3)
    pp2 = torch.diagonal(pp, offset=0, dim1=0, dim2=1).clone()  # (N, 3)

    directions = pp2 - centers                                  # (N, 3)
    r = F.normalize(directions, dim=-1, eps=1e-8)
    return r

def compute_axis_dispersion_ratio(r: Tensor) -> float:
    """
    Compute Camera Axis Dispersion Ratio (CADR).
    r: (N, 3) unit direction vectors.
    Returns:
        ratio in [0, 2], higher indicates a more isotropic distribution.
    """
    if not (r.ndim == 2 and r.shape[1] == 3):
        raise ValueError("r must have shape (N, 3)")
    if r.shape[0] < 2:
        raise ValueError("r must contain at least 2 vectors")

    r_centered = r - r.mean(dim=0, keepdim=True)
    cov = (r_centered.T @ r_centered) / (r.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov.double()).sort(descending=True).values.float()
    eps = 1e-8
    ratio = (eigvals[1] + eigvals[2]) / (eigvals[0] + eps)
    return float(ratio.item())

def measure_camera_distribution(cameras) -> float:
    """Return CADR ratio as a float."""
    r_vec = compute_optical_axis_directions(cameras)
    return compute_axis_dispersion_ratio(r_vec)

