import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.carla_env import CarlaEnv
from agents.feature_extractor import MultiModalExtractor
from training.callbacks import CheckpointCallback


def main():
    with open("configs/training.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("logs", exist_ok=True)
    env = DummyVecEnv([lambda: Monitor(CarlaEnv(), filename="logs/monitor.csv")])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_obs_keys=["state"],
        norm_reward=True,
        clip_obs=10.0,
    )

    policy_kwargs = dict(
        features_extractor_class=MultiModalExtractor,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        tensorboard_log="logs/",
        verbose=1,
        policy_kwargs=policy_kwargs,
        **cfg["ppo"],
    )

    model.learn(
        total_timesteps=cfg["total_steps"],
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
