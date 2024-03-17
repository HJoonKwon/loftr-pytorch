import numpy as np
import cv2


# https://github.com/zju3dv/LoFTR/blob/df7ca80f917334b94cfbe32cc2901e09a80e70a8/src/utils/metrics.py#L72
def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC
    )
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def project_points(points_3d, R, t, K):
    assert points_3d.shape[1] == 3 and len(points_3d.shape) == 2
    assert R.shape == (3, 3) and np.allclose(np.linalg.det(R), 1)
    assert t.shape == (3,) or t.shape == (3, 1)
    assert K.shape == (3, 3)
    if len(t.shape) == 1:
        t = t[..., None]
    points_2d = (K @ (R @ points_3d.T + t)).T  # (N, 3)
    points_2d /= points_2d[:, [2]]  # (N, 3) / (N, 1)  = (N, 3)
    return points_2d[:, :2]


def compute_rodrigues_angle(R):
    assert R.shape == (3, 3)
    assert np.allclose(np.linalg.det(R), 1)
    theta = np.arccos((np.trace(R) - 1) / 2)
    return theta


def compute_rotation_error(R_true, R_est):
    assert R_true.shape == (3, 3)
    assert R_est.shape == (3, 3)
    assert np.allclose(np.linalg.det(R_true), 1)
    assert np.allclose(np.linalg.det(R_est), 1)
    err_R = R_est.T @ R_true
    err_angle = compute_rodrigues_angle(err_R)
    return err_angle


def compute_translation_error(t_gt, t):
    assert t_gt.shape[0] == 3
    assert t.shape[0] == 3
    assert t.size == t_gt.size
    t = np.squeeze(t)
    t_gt = np.squeeze(t_gt)
    cos_theta = (t.T @ t_gt) / np.linalg.norm(t) / np.linalg.norm(t_gt)
    err_angle = np.arccos(np.clip(cos_theta, -1, 1))
    err_angle = np.minimum(err_angle, np.pi - err_angle)
    return err_angle
