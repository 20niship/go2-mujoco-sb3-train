# convert quat to euler angles using scipy

from scipy.spatial.transform import Rotation as R


def quat_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Parameters:
    quat (list or np.ndarray): Quaternion in the form [x, y, z, w].

    Returns:
    np.ndarray: Euler angles in radians.
    """
    r = R.from_quat(quat)
    return r.as_euler(
        "xyz", degrees=False
    )  # Change 'xyz' to 'zyx' if needed for different convention
