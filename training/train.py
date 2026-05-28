import argparse
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
from agents.feature_extractor import MultiModalExtractor
from training.callbacks import CheckpointCallback


CHECKPOINT_DIR = "checkpoints"
VECNORM_FILE = "vecnormalize.pkl"


def load_phase(phase_name):
    with open("configs/curriculum.yaml") as f:
        phases = yaml.safe_load(f)["phases"]
    for p in phases:
        if p["name"] == phase_name:
            return p
    names = [p["name"] for p in phases]
    raise SystemExit(f"Phase '{phase_name}' not found. Available: {names}")


def find_latest_checkpoint():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "model_*.zip"))
    if not files:
        return None, 0
    def step_of(path):
        m = re.search(r"model_(\d+)\.zip$", path.replace("\\", "/"))
        return int(m.group(1)) if m else -1
    latest = max(files, key=step_of)
    return latest, step_of(latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, help="Phase name from curriculum.yaml (e.g. p1)")
    args = parser.parse_args()

    phase = load_phase(args.phase)
    print(f"Phase {phase['name']}: town={phase['town']} weather={phase['weather']} "
          f"traffic={phase['traffic']} target_step={phase['target_step']:,}")

    with open("configs/training.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    ckpt_path, ckpt_step = find_latest_checkpoint()

    remaining = phase["target_step"] - ckpt_step
    if remaining <= 0:
        print(f"Latest checkpoint is at step {ckpt_step:,}, already past target "
              f"{phase['target_step']:,}. Nothing to do.")
        return

    def make_env():
        return Monitor(
            CarlaEnv(
                global_step=ckpt_step,
                town=phase["town"],
                weather=phase["weather"],
                traffic_count=phase["traffic"],
            ),
            filename="logs/monitor.csv",
        )

    raw_env = DummyVecEnv([make_env])

    vecnorm_path = os.path.join(CHECKPOINT_DIR, VECNORM_FILE)
    if ckpt_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, raw_env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(
            raw_env,
            norm_obs=True,
            norm_obs_keys=["state"],
            norm_reward=True,
            clip_obs=10.0,
        )

    if ckpt_path:
        print(f"Resuming PPO from {ckpt_path} (step {ckpt_step:,})")
        model = PPO.load(ckpt_path, env=env, tensorboard_log="logs/")
    else:
        print("No checkpoint found, starting fresh PPO")
        policy_kwargs = dict(features_extractor_class=MultiModalExtractor)
        model = PPO(
            "MultiInputPolicy",
            env,
            tensorboard_log="logs/",
            verbose=1,
            policy_kwargs=policy_kwargs,
            **cfg["ppo"],
        )

    print(f"Training {remaining:,} steps (until step {phase['target_step']:,})")

    callback = CheckpointCallback(cfg["checkpoint_freq"], CHECKPOINT_DIR)

    try:
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=False,
            callback=callback,
            tb_log_name="PPO",
        )
        print(f"Reached target step {phase['target_step']:,} for phase {phase['name']}.")
    except KeyboardInterrupt:
        print("\nCtrl+C received, saving before exit...")
    finally:
        save_step = model.num_timesteps
        model.save(os.path.join(CHECKPOINT_DIR, f"model_{save_step}"))
        env.save(vecnorm_path)
        print(f"Saved checkpoint at step {save_step:,}")


if __name__ == "__main__":
    main()
