from stable_baselines3 import SAC
from env.go2_flat_env import MultiGo2Env
from env.utils import LearningConfig
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="事前学習済みモデル。指定がない場合は新規に学習開始",
    )
    args = parser.parse_args()

    cfg = LearningConfig()
    cfg.gui = True
    cfg.log_wandb = False

    env = MultiGo2Env(cfg)
    # check_env(env)  # Gym API に準拠しているか確認
    model = SAC.load(
        args.model,
        env=env,
        verbose=1,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
    )

    obs, _ = env.reset()

    total_timesteps = 1000000
    for _ in range(total_timesteps):
        action, _ = model.predict(obs)
        obs, reward, done,  _, info = env.step(action)


if __name__ == "__main__":
    main()
