import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from env.go2_flat_env import MultiGo2Env
from env.utils import LearningConfig

# Initialize the environment

cfg = LearningConfig()
cfg.gui = True

env = MultiGo2Env(cfg)
# check_env(env)  # Gym API に準拠しているか確認

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=cfg.learning_rate,
    buffer_size=cfg.buffer_size,
)
model.learn(total_timesteps=cfg.total_timesteps)

model.save("multi_go2_ppo")
