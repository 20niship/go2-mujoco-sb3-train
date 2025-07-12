from dataclasses import dataclass
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


# sac config
@dataclass
class LearningConfig:
    """
    Configuration for the learning process
    """

    policy: str = "MlpPolicy"
    total_timesteps: int = 2000000
    buffer_size: int = 1000000
    learning_rate: float = 0.001
    gui = False
