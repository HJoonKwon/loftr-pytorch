import torch


def inverse_transformation(T):
    """
    Invert a 4x4 transformation matrix [R | t; 0 0 0 1]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.transpose(1, 0)
    t_inv = -R_inv @ t
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv
