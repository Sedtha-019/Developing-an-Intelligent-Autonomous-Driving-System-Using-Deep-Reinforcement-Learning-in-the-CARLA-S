import argparse
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.running_mean_std import RunningMeanStd
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reward-changed", action="store_true",
        help="Soft recovery after a reward-function change: reset "
             "VecNormalize reward stats (keep obs stats) and apply a modest "
             "ent_coef bump (0.005 -> 0.008). Value head is kept so the "
             "policy's advantage signal stays meaningful. Backs up the "
             "existing vecnormalize.pkl to "
             "vecnormalize_pre_reward_change.pkl.",
    )
    p.add_argument(
        "--reset-vf", action="store_true",
        help="ADDITIONAL aggressive option for --reward-changed: also "
             "re-initialise the value head. Only use this if soft recovery "
             "fails to escape a bad attractor after a few hundred K steps. "
             "Tends to make the policy worse for ~50-100K steps as vf "
             "relearns.",
    )
    p.add_argument(
        "--ent-coef", type=float, default=None,
        help="Override ent_coef. Defaults: keep model's value, or 0.008 "
             "when --reward-changed is set.",
    )
    return p.parse_args()


def reset_value_head(model):
    vnet = model.policy.value_net
    nn.init.orthogonal_(vnet.weight, gain=1.0)
    nn.init.zeros_(vnet.bias)


def reset_reward_norm(env):
    env.ret_rms = RunningMeanStd(shape=())
    env.returns = np.zeros(env.num_envs)


def main():
    args = parse_args()

    with open("configs/training.yaml") as f:
        cfg = yaml.safe_load(f)

    ckpt = find_latest_checkpoint("checkpoints")
    print(f"Resuming from: {ckpt}")
    model = PPO.load(ckpt)
    model.tensorboard_log = "logs/"

    if args.reward_changed:
        if args.reset_vf:
            print("[reward-changed] Re-initialising value head (--reset-vf).")
            reset_value_head(model)
        else:
            print("[reward-changed] Keeping value head (soft recovery).")
        ent_coef = args.ent_coef if args.ent_coef is not None else 0.008
        model.ent_coef = ent_coef
        print(f"[reward-changed] ent_coef = {ent_coef}")
    elif args.reset_vf:
        print("--reset-vf is only valid with --reward-changed; ignoring.")
    elif args.ent_coef is not None:
        model.ent_coef = args.ent_coef
        print(f"ent_coef override = {args.ent_coef}")

    os.makedirs("logs", exist_ok=True)
    env = DummyVecEnv([
        lambda: Monitor(
            CarlaEnv(global_step=model.num_timesteps),
            filename="logs/monitor.csv",
        )
    ])

    vecnorm_path = "checkpoints/vecnormalize.pkl"
    if args.reward_changed and os.path.exists(vecnorm_path):
        backup = "checkpoints/vecnormalize_pre_reward_change.pkl"
        shutil.copy2(vecnorm_path, backup)
        print(f"[reward-changed] Backed up VecNormalize → {backup}")

    env = VecNormalize.load(vecnorm_path, env)
    env.training = True
    env.norm_reward = True

    if args.reward_changed:
        print("[reward-changed] Resetting reward-normalisation running stats "
              "(observation stats kept).")
        reset_reward_norm(env)

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
