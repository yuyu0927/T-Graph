"""
Adapted from code originally written by David Novotny.
"""
import torch
from pytorch3d.transforms import Rotate, Translate


def intersect_skew_line_groups(p, r, mask):
    # p, r both of shape (B, N, n_intersected_lines, 3)
    # mask of shape (B, N, n_intersected_lines)
    p_intersect, r = intersect_skew_lines_high_dim(p, r, mask=mask)
    _, p_line_intersect = _point_line_distance(
        p, r, p_intersect[..., None, :].expand_as(p)
    )
    intersect_dist_squared = ((p_line_intersect - p_intersect[..., None, :]) ** 2).sum(
        dim=-1
    )
    return p_intersect, p_line_intersect, intersect_dist_squared, r


def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    if torch.any(torch.isnan(p_intersect)):
        print(p_intersect)
        assert False
    return p_intersect, r


def _point_line_distance(p1, r1, p2):
    df = p2 - p1
    proj_vector = df - ((df * r1).sum(dim=-1, keepdim=True) * r1)
    line_pt_nearest = p2 - proj_vector
    d = (proj_vector).norm(dim=-1)
    return d, line_pt_nearest


def compute_optical_axis_intersection(cameras):
    centers = cameras.get_camera_center()
    principal_points = cameras.principal_point

    one_vec = torch.ones((len(cameras), 1))
    optical_axis = torch.cat((principal_points, one_vec), -1)

    pp = cameras.unproject_points(optical_axis, from_ndc=True, world_coordinates=True)

    pp2 = torch.zeros((pp.shape[0], 3))
    for i in range(0, pp.shape[0]):
        pp2[i] = pp[i][i]

    directions = pp2 - centers
    centers = centers.unsqueeze(0).unsqueeze(0)
    directions = directions.unsqueeze(0).unsqueeze(0)

    p_intersect, p_line_intersect, _, r = intersect_skew_line_groups(
        p=centers, r=directions, mask=None
    )

    p_intersect = p_intersect.squeeze().unsqueeze(0)
    dist = (p_intersect - centers).norm(dim=-1)

    return p_intersect, dist, p_line_intersect, pp2, r


def normalize_cameras(cameras, scale=1.0):
    """
    Normalizes cameras such that the optical axes point to the origin and the average
    distance to the origin is 1.

    Args:
        cameras (List[camera]).
    """

    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()
    new_transform = new_cameras.get_world_to_view_transform()

    p_intersect, dist, p_line_intersect, pp, r = compute_optical_axis_intersection(
        cameras
    )
    t = Translate(p_intersect)

    scale = dist.squeeze()[0]

    # Degenerate case
    if scale == 0:
        print(cameras.T)
        print(new_transform.get_matrix()[:, 3, :3])
        return -1
    assert scale != 0

    new_transform = t.compose(new_transform)
    new_cameras.R = new_transform.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform.get_matrix()[:, 3, :3] / scale
    return new_cameras, p_intersect, p_line_intersect, pp, r

def normalize_cameras_1(cameras, scale: float = 1.0):
    """
    Normalize a batch of cameras by translating the scene so that the estimated
    optical-axis intersection is at the origin.

    Notes:
      - This function currently DOES NOT apply scaling to translations. The `scale`
        argument is only checked for degeneracy, but not used to rescale T.
      - We return `scale_max` (max distance to the intersection) for optional external
        normalization.

    Args:
        cameras: A PyTorch3D Cameras object (e.g., PerspectiveCameras / FoVPerspectiveCameras).
        scale:  A non-zero scalar used only as a degeneracy guard (kept for compatibility).

    Returns:
        new_cameras: normalized cameras (same type as input)
        p_intersect: (3,) intersection point in world coordinates
        p_line_intersect, pp, r: auxiliary outputs from compute_optical_axis_intersection
        scale_max: max distance-to-intersection among cameras (float)
    """
    # Clone to avoid mutating the input cameras in-place
    new_cameras = cameras.clone()

    # Estimate where optical axes intersect + per-camera distances to that point
    p_intersect, dist, p_line_intersect, pp, r = compute_optical_axis_intersection(cameras)

    # Degenerate guard (kept for backward-compatibility)
    if scale == 0:
        # Helpful debug prints
        print(cameras.T)
        print(new_cameras.get_world_to_view_transform().get_matrix()[:, 3, :3])
        return -1

    # Robust scalar summary of distances (used externally if desired)
    dist_flat = dist.reshape(-1)
    scale_max = float(dist_flat.max().item()) if dist_flat.numel() > 0 else 0.0

    # Translate world so that p_intersect maps to the origin in the new frame
    # (PyTorch3D uses row-major transform matrices; world->view translation lives in row 3, cols :3)
    transform = new_cameras.get_world_to_view_transform()
    transform = Translate(p_intersect).compose(transform)

    mat = transform.get_matrix()  # (N, 4, 4)
    new_cameras.R = mat[:, :3, :3]
    new_cameras.T = mat[:, 3, :3]  # no scaling here; caller can divide by scale_max if needed

    return new_cameras, p_intersect, p_line_intersect, pp, r, scale_max


def first_camera_transform(cameras, rotation_only=False):
    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()
    new_transform = new_cameras.get_world_to_view_transform()
    tR = Rotate(new_cameras.R[0].unsqueeze(0))
    if rotation_only:
        t = tR.inverse()
    else:
        tT = Translate(new_cameras.T[0].unsqueeze(0))
        t = tR.compose(tT).inverse()

    new_transform = t.compose(new_transform)
    new_cameras.R = new_transform.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform.get_matrix()[:, 3, :3]

    return new_cameras
