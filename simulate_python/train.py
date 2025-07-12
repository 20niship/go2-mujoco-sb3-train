import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.go2_flat_env import MultiGo2Env

env = MultiGo2Env()
# check_env(env)  # Gym API に準拠しているか確認

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=100000)

model.save("multi_go2_ppo")
