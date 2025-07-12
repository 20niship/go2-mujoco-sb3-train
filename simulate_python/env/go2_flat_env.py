import numpy as np
from gymnasium.spaces import Box
import random
from gymnasium.envs.mujoco import MujocoEnv
import config
from .utils import quat_to_euler, LearningConfig
import wandb


class MultiGo2Env(MujocoEnv):
    OBS_SHAPE = 45  # 観測空間の形状
    NUM_JOINTS = 12  # 1体の関節数（例）

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            # "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, cfg: LearningConfig):
        obs_space = Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_SHAPE,), dtype=np.float64
        )
        xml_file = config.ROBOT_SCENE

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=obs_space,
        )

        self.num_agents = 1  # 複数体のロボットを扱う
        self.nsteps = 0

        self.observation_space = obs_space
        self.cfg = cfg
        self.render_mode = "human" if cfg.gui else None
        self.action_space = Box(
            low=-2,
            high=2,
            shape=(self.num_agents * self.NUM_JOINTS,),
            dtype=np.float32,
        )
        self.prev_action = np.zeros(self.num_agents * self.NUM_JOINTS, dtype=np.float32)
        # self.observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(self.num_agents * 48,), dtype=np.float32
        # )
        # self.action_space = Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(self.num_agents * self.agent_joint_num,),
        #     dtype=np.float32,
        # )
        self.cmd_vel = [1, 0, 0]

        self._reset_noise_scale = 0.02  # リセット時のノイズスケール

        self.jpos = np.zeros(self.NUM_JOINTS, dtype=np.float32)  # 各関節の位置
        self.jvel = np.zeros(self.NUM_JOINTS, dtype=np.float32)  # 各関節の速度
        self.force = np.zeros(self.NUM_JOINTS, dtype=np.float32)  # 各関節のトルク
        self.imu_acc = np.array([0.0, 0.0, -9.8], dtype=np.float32)  # IMUの加速度
        self.imu_gyro = np.zeros(3, dtype=np.float32)  # IMUの角速度
        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.rot_euler = np.zeros(3, dtype=np.float32)  # オイラー角

    def reset(self, seed=None, options=None):
        # self.cmd_vel = [
        #     random.uniform(-1.0, 1.0),  # x方向の速度
        #     random.uniform(-1.0, 1.0),  # y方向の速度
        #     random.uniform(-1.0, 1.0),  # 角速度
        # ]
        self.reset_model()
        self.nsteps = 0
        obs = self._get_obs()

        return obs, {}

    def terminated(self):
        # ここでは、ロボットが倒れたかどうかをチェックする
        # 例えば、ロボットの高さが一定以下になったら終了
        base_height = self.get_body_com("base_link")[2]
        euler = self.rot_euler * 180 / np.pi  # ラジアンから度に変換
        return base_height < 0.2 or abs(euler[1]) > 45 or abs(euler[2]) > 45

    @property
    def contact_cost(self):
        # 接触力に基づいてコストを計算
        contact_forces = self.data.cfrc_ext
        contact_cost = np.sum(np.square(contact_forces)) * 0.001
        return contact_cost

    def _do_action_pid(self, action: np.ndarray):
        Kp = 60.0  # 比例ゲイン
        Kd = 5.0  # 微分ゲイン

        Kp = 0.0  # 比例ゲイン
        Kd = 0.0  # 微分ゲイン
        error = action - self.jpos
        torque = Kp * error - Kd * self.jvel
        self.do_simulation(torque, self.frame_skip)

    def _get_xy_pos(self):
        # ロボットの位置を取得
        xy_pos = self.get_body_com("base_link")[:2].copy()  # x, y座標を取得
        return xy_pos

    def step(self, action: np.ndarray):
        action = action.flatten()
        action = np.zeros_like(action)
        xy_pos_before = self._get_xy_pos()
        self._do_action_pid(action)

        if self.render_mode == "human":
            self.render()

        xy_pos_after = self._get_xy_pos()

        xy_velocity = (xy_pos_after - xy_pos_before) / self.dt
        x_vel, y_vel = xy_velocity

        forward_rew = x_vel - y_vel
        ctrl_cost = np.sum(np.square(action)) * 0.01

        terminated = self.terminated()

        healthy_reward = 0.2
        if terminated:
            healthy_reward = -10.0

        imu_cost = -0.02 * (np.linalg.norm(self.imu_gyro))

        reward = forward_rew + healthy_reward - ctrl_cost + imu_cost

        obs = self._get_obs()
        info = {
            "info/rew_forward": forward_rew,
            "info/ctrl_cost": -ctrl_cost,
            "info/rew_survive": healthy_reward,
            "info/rew_imu": imu_cost,
            "info/x_position": xy_pos_after[0],
            "info/y_position": xy_pos_after[1],
        }

        if self.cfg.log_wandb and self.nsteps % self.cfg.wandb_log_steps == 0:
            wandb.log(info)  # type: ignore

        # if self._use_contact_forces:
        #     contact_cost = self.contact_cost
        #     costs += contact_cost
        #     info["reward_ctrl"] = -contact_cost

        self.prev_action = action.copy()  # 前のアクションを保存

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, terminated, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        sensor = self.data.sensordata

        self.jpos = sensor[0:12]  # 各関節の位置
        self.jvel = sensor[12:24]  # 各関節の速度
        self.force = sensor[24:36]  # 各関節のトルク
        self.imu_quat = sensor[36:40]  # IMUのクォータニオン
        self.imu_gyro = sensor[40:43]
        self.imu_acc = sensor[43:46]
        self.frame_pos = sensor[46:49]
        self.frame_vel = sensor[49:52]
        if np.linalg.norm(self.imu_quat) == 0:
            self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            self.rot_euler = quat_to_euler(
                self.imu_quat
            )  # クォータニオンからオイラー角に変換

        # contact_force = self.contact_forces.flat.copy()

        # if self._exclude_current_positions_from_observation:
        #     position = position[2:]

        obs = np.concatenate(
            [
                self.imu_acc,
                # imu_quat,
                self.imu_gyro,
                self.cmd_vel,  # コマンド速度
                self.jpos,
                self.jvel,
                self.prev_action,  # 前のアクション
            ]
        )
        assert len(obs) == self.OBS_SHAPE, (
            f"Observation shape mismatch {len(obs)} != {self.OBS_SHAPE}"
        )

        return obs
        # 各ロボットのセンサ情報を結合
        # obs = []
        # for i in range(self.num_agents):
        #     base_pos = self.sim.data.get_body_xpos(f"go2_{i}_base")
        #     joint_qpos = self.sim.data.qpos[
        #         i * 19 : i * 19 + 12
        #     ]  # 各ロボットの関節角度
        #     obs.append(np.concatenate([base_pos, joint_qpos]))
        return np.concatenate(obs)

    def _compute_reward(self):
        # シンプルに前進距離の合計を報酬に
        reward = 0.0
        for i in range(self.num_agents):
            base_x = self.sim.data.get_body_xpos(f"go2_{i}_base")[0]
            reward += base_x
        return reward

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        observation = self._get_obs()

        return observation
