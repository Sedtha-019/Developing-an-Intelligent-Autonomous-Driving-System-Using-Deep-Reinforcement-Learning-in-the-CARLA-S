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
from training.callbacks import CheckpointCallback


PHASE2_CKPT_DIR = "checkpoints_phase2"
PHASE2_LOG_DIR = "logs_phase2"

SEED_MODEL = "checkpoints/phase1_final"
SEED_VECNORM = "checkpoints/vecnormalize_phase1.pkl"


def _step_of(path):
    m = re.search(r"model_(\d+)\.zip$", path)
    return int(m.group(1)) if m else -1


def resolve_checkpoint():
    candidates = glob.glob(os.path.join(PHASE2_CKPT_DIR, "model_*.zip"))
    if candidates:
        latest = max(candidates, key=_step_of)
        model_path = latest[:-4]
        vecnorm_path = os.path.join(PHASE2_CKPT_DIR, "vecnormalize.pkl")
        if not os.path.exists(vecnorm_path):
            raise FileNotFoundError(
                f"Phase 2 checkpoint found but {vecnorm_path} missing"
            )
        return model_path, vecnorm_path, True

    if not os.path.exists(SEED_MODEL + ".zip"):
        raise FileNotFoundError(
            f"Phase 1 seed missing: {SEED_MODEL}.zip "
            "(run Phase 1 first, then copy model_1000000.zip to phase1_final.zip)"
        )
    if not os.path.exists(SEED_VECNORM):
        raise FileNotFoundError(f"Phase 1 VecNormalize seed missing: {SEED_VECNORM}")
    return SEED_MODEL, SEED_VECNORM, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curriculum",
        default="configs/curriculum_phase2_town01.yaml",
        help="Path to a single-town curriculum YAML",
    )
    args = parser.parse_args()

    with open("configs/training_phase2.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(PHASE2_CKPT_DIR, exist_ok=True)
    os.makedirs(PHASE2_LOG_DIR, exist_ok=True)

    model_path, vecnorm_path, is_resume = resolve_checkpoint()
    print(f"[phase2] {'resuming' if is_resume else 'seeding from phase 1'}")
    print(f"[phase2] model: {model_path}")
    print(f"[phase2] vecnormalize: {vecnorm_path}")
    print(f"[phase2] curriculum: {args.curriculum}")

    model = PPO.load(model_path)
    model.tensorboard_log = PHASE2_LOG_DIR

    curriculum_path = args.curriculum

    env = DummyVecEnv([
        lambda: Monitor(
            CarlaEnv(
                global_step=model.num_timesteps,
                curriculum_path=curriculum_path,
            ),
            filename=os.path.join(PHASE2_LOG_DIR, "monitor.csv"),
        )
    ])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = True
    env.norm_reward = True

    model.set_env(env)

    remaining = max(cfg["total_steps"] - model.num_timesteps, 0)
    print(f"[phase2] current num_timesteps={model.num_timesteps}, remaining={remaining}")

    if remaining == 0:
        print("[phase2] total_steps already reached; raise it in configs/training_phase2.yaml")
        return

    model.learn(
        total_timesteps=remaining,
        reset_num_timesteps=False,
        callback=CheckpointCallback(cfg["checkpoint_freq"], PHASE2_CKPT_DIR),
    )

    model.save(os.path.join(PHASE2_CKPT_DIR, "phase2_final"))
    env.save(os.path.join(PHASE2_CKPT_DIR, "vecnormalize.pkl"))


if __name__ == "__main__":
    main()
