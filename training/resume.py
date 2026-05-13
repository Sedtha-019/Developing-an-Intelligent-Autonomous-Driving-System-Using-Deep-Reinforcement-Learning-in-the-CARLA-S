import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.carla_env import CarlaEnv
from training.callbacks import CheckpointCallback


def find_latest_checkpoint(ckpt_dir):
    if os.path.exists(os.path.join(ckpt_dir, "latest.zip")):
        return os.path.join(ckpt_dir, "latest")

    candidates = glob.glob(os.path.join(ckpt_dir, "model_*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    def step_of(path):
        m = re.search(r"model_(\d+)\.zip$", path)
        return int(m.group(1)) if m else -1

    latest = max(candidates, key=step_of)
    return latest[:-4]


def main():
    with open("configs/training.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt = find_latest_checkpoint("checkpoints")
    print(f"Resuming from: {ckpt}")
    model = PPO.load(ckpt)
    model.tensorboard_log = "logs/"

    os.makedirs("logs", exist_ok=True)
    env = DummyVecEnv([
        lambda: Monitor(
            CarlaEnv(global_step=model.num_timesteps),
            filename="logs/monitor.csv",
        )
    ])
    env = VecNormalize.load("checkpoints/vecnormalize.pkl", env)
    env.training = True
    env.norm_reward = True

    model.set_env(env)

    remaining = max(cfg["total_steps"] - model.num_timesteps, 0)

    model.learn(
        total_timesteps=remaining,
        reset_num_timesteps=False,
        callback=CheckpointCallback(
            cfg["checkpoint_freq"],
            "checkpoints",
        ),
    )

    model.save("checkpoints/latest")
    env.save("checkpoints/vecnormalize.pkl")


if __name__ == "__main__":
    main()
