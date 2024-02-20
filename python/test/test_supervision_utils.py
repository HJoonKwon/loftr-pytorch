import torch
from loftr_pytorch.supervision.utils import inverse_transformation


def test_inverse_transformation():
    T = torch.tensor(
        [
            [-0.77012685, -0.03038916, -0.63716648, -0.997712],
            [0.20635993, 0.93328359, -0.29393421, -0.0395286],
            [0.60358944, -0.35785226, -0.71247565, 0.716545],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.double,
    )
    T_inv = inverse_transformation(T)
    T_inv_expected = torch.linalg.inv(T)
    assert torch.allclose(T_inv, T_inv_expected)
