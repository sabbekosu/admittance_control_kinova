import numpy as np


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_from_rotvec(r):
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = r / angle
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])


def rotvec_from_quat(q):
    q = q if q[3] >= 0 else -q  # shortest rotation
    s = np.linalg.norm(q[:3])
    if s < 1e-12:
        return np.zeros(3)
    return q[:3] / s * 2 * np.arctan2(s, q[3])
