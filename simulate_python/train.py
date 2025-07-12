from stable_baselines3 import SAC
import sys

# from stable_baselines3.common.env_checker import check_env
from env.go2_flat_env import MultiGo2Env
from env.utils import LearningConfig

from wandb.integration.sb3 import WandbCallback  # type: ignore
import wandb  # type: ignore

cfg = LearningConfig()


run_id = "go2-flat"
if len(sys.argv) > 1:
    run_id = sys.argv[1]

run = wandb.init(
    project="unitree-go2-learning",
    config=cfg,
    name=run_id,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

cfg.gui = False

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

wandb_callback = WandbCallback(
    model_save_freq=1000,
    gradient_save_freq=1000,
    model_save_path=f"models/{run_id}",
    verbose=2,
)
model.learn(total_timesteps=cfg.total_timesteps)

model.save("multi_go2_ppo")
