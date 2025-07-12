from stable_baselines3 import SAC
import random

from env.go2_flat_env import MultiGo2Env
from env.utils import LearningConfig

from wandb.integration.sb3 import WandbCallback  # type: ignore
import wandb  # type: ignore

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="事前学習済みモデル。指定がない場合は新規に学習開始",
    )
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument(
        "--wandb", type=bool, default=True, help="Wandbでのログを有効にするか"
    )

    args = parser.parse_args()

    cfg = LearningConfig()
    cfg.gui = args.gui
    cfg.log_wandb = args.wandb

    run_id = f"{random.randint(0, 1000000)}"

    if cfg.log_wandb:
        run = wandb.init(  # type: ignore
            project="unitree-go2-learning",
            config=cfg,
            # name=args.run_id,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        run_id = run.id

    env = MultiGo2Env(cfg)
    # check_env(env)  # Gym API に準拠しているか確認

    if args.base_model:
        # 既存モデルから再学習
        print(f"Loading base model from {args.base_model}")
        model = SAC.load(
            args.base_model,
            env=env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
        )

    callbacks = []
    if cfg.log_wandb:
        wandb_callback = WandbCallback(
            model_save_freq=1000,
            gradient_save_freq=1000,
            model_save_path=f"models/{run_id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)

    model.save("multi_go2_ppo")


if __name__ == "__main__":
    main()
