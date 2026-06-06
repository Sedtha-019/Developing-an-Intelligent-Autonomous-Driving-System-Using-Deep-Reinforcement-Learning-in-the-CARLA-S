"""
Collect visual samples of what the agent sees during a run in CARLA.

Saves per-step composite images (raw RGB + 4 grayscale frames + annotations)
and a state_log.csv. Does NOT modify any training code.

Usage
-----
  # drive with CARLA autopilot
  python tools/collect_samples.py --phase p1 --mode autopilot --steps 200

  # drive with a trained PPO checkpoint
  python tools/collect_samples.py --phase p1 --mode checkpoint --steps 200

  # choose a specific checkpoint
  python tools/collect_samples.py --phase p1 --mode checkpoint --steps 200 \
      --checkpoint checkpoints/model_2000000.zip

Output
------
  logs/samples/<timestamp>/
      step_000001.png   composite image (raw | frame t-3 | t-2 | t-1 | t) + text bar
      step_000002.png
      ...
      state_log.csv     one row per step: step, speed, lateral, heading, steer,
                        npc_dist, npc_rel_speed, action_steer, action_thr,
                        action_brk, reward, terminated
"""

import argparse
import csv
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import yaml

from env.carla_env import CarlaEnv

CHECKPOINT_DIR = "checkpoints"
VECNORM_FILE   = os.path.join(CHECKPOINT_DIR, "vecnormalize.pkl")
OUTPUT_ROOT    = "logs/samples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_phase(phase_name: str) -> dict:
    with open("configs/curriculum.yaml") as f:
        phases = yaml.safe_load(f)["phases"]
    for p in phases:
        if p["name"] == phase_name:
            return p
    names = [ph["name"] for ph in phases]
    raise SystemExit(f"Phase '{phase_name}' not found. Available: {names}")


def find_latest_checkpoint() -> str | None:
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "model_*.zip"))
    if not files:
        return None
    def step_of(path):
        m = re.search(r"model_(\d+)\.zip$", path.replace("\\", "/"))
        return int(m.group(1)) if m else -1
    return max(files, key=step_of)


def upscale_gray(frame_1ch: np.ndarray, size: int = 128) -> np.ndarray:
    """Upscale a single 84×84 float [0,1] frame to `size`×`size` uint8 BGR."""
    f = (frame_1ch * 255).clip(0, 255).astype(np.uint8)
    f = cv2.resize(f, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)


def build_composite(raw_bgr: np.ndarray, stacked: np.ndarray,
                    state: np.ndarray, action: np.ndarray,
                    reward: float, step: int) -> np.ndarray:
    """
    Build a composite image:
      [raw 128×128] [frame t-3] [frame t-2] [frame t-1] [frame t]
      + a text annotation bar at the bottom.

    raw_bgr  : (128,128,3) uint8
    stacked  : (4,84,84)   float32 [0,1]
    state    : (6,)        float32
    action   : (3,)        float32
    """
    SIZE = 128
    panels = [raw_bgr]
    for i in range(4):
        panels.append(upscale_gray(stacked[i], SIZE))

    canvas = np.hstack(panels)   # (128, 640, 3)

    # annotation bar (black background, white text)
    bar_h = 36
    bar = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)

    spd, lat, hdg, prv_steer, npc_d, npc_rel = state
    a_steer, a_thr, a_brk = action

    line1 = (f"step={step:05d}  "
             f"spd={spd:.2f}m/s  lat={lat:.3f}m  hdg={hdg:.3f}rad  "
             f"npc={npc_d:.1f}m  rel={npc_rel:.2f}m/s")
    line2 = (f"action: steer={a_steer:+.3f}  thr={a_thr:.3f}  brk={a_brk:.3f}   "
             f"reward={reward:+.3f}")

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    color = (220, 220, 220)
    cv2.putText(bar, line1, (4, 13), font, scale, color, 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (4, 29), font, scale, color, 1, cv2.LINE_AA)

    # column labels above the panels
    label_bar = np.zeros((18, canvas.shape[1], 3), dtype=np.uint8)
    labels = ["RAW 128x128 RGB", "frame t-3 (oldest)", "frame t-2",
              "frame t-1", "frame t (newest)"]
    for i, lbl in enumerate(labels):
        cv2.putText(label_bar, lbl, (i * SIZE + 4, 13),
                    font, 0.33, (180, 180, 60), 1, cv2.LINE_AA)

    return np.vstack([label_bar, canvas, bar])


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect(args):
    phase = load_phase(args.phase)
    tag   = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"{tag}_{args.phase}_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output → {out_dir}")

    env = CarlaEnv(
        global_step=0,
        town=phase["town"],
        weather=phase["weather"],
        traffic_count=phase["traffic"] if args.mode != "autopilot" else 0,
    )

    # load PPO model if checkpoint mode
    model = None
    if args.mode == "checkpoint":
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from agents.feature_extractor import MultiModalExtractor

            ckpt = args.checkpoint or find_latest_checkpoint()
            if ckpt is None:
                raise SystemExit("No checkpoint found in checkpoints/. "
                                 "Use --mode autopilot or supply --checkpoint <path>.")
            print(f"Loading checkpoint: {ckpt}")

            # wrap env so VecNormalize can normalise the state vector
            vec_env = DummyVecEnv([lambda: env])
            if os.path.exists(VECNORM_FILE):
                vec_env = VecNormalize.load(VECNORM_FILE, vec_env)
                vec_env.training = False
                vec_env.norm_reward = False
            else:
                vec_env = VecNormalize(vec_env, norm_obs_keys=["state"],
                                       norm_reward=False, training=False)

            policy_kwargs = dict(features_extractor_class=MultiModalExtractor,
                                 features_extractor_kwargs=dict(features_dim=256))
            model = PPO.load(ckpt, env=vec_env, device="cpu",
                             custom_objects={"policy_kwargs": policy_kwargs})
            print("Checkpoint loaded.")
            use_vecenv = True
        except Exception as exc:
            print(f"[WARN] Could not load checkpoint ({exc}). Falling back to autopilot.")
            model = None
            use_vecenv = False
    else:
        use_vecenv = False

    # CSV log
    csv_path = os.path.join(out_dir, "state_log.csv")
    csv_fields = ["step", "speed_ms", "lateral_m", "heading_rad",
                  "prev_steer", "npc_dist_m", "npc_rel_speed",
                  "action_steer", "action_thr", "action_brk",
                  "reward", "terminated"]
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    total = 0
    saved = 0

    try:
        obs, _ = env.reset()
        if use_vecenv:
            vec_obs, _ = vec_env.reset()

        while saved < args.steps:
            # --- choose action ---
            if args.mode == "autopilot":
                env.vehicle.set_autopilot(True)
                action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            elif model is not None:
                vec_action, _ = model.predict(vec_obs, deterministic=True)
                action = vec_action[0]
            else:
                action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            obs, reward, terminated, truncated, _ = env.step(action)
            total += 1

            if use_vecenv and model is not None:
                vec_obs, _, _, _, _ = vec_env.step(vec_action)

            # --- grab raw camera frame (BGR) ---
            if env.image is None:
                if terminated or truncated:
                    obs, _ = env.reset()
                    if use_vecenv:
                        vec_obs, _ = vec_env.reset()
                continue

            import numpy as _np
            raw_arr = _np.frombuffer(env.image.raw_data, dtype=np.uint8)
            raw_bgr = raw_arr.reshape(
                (env.image.height, env.image.width, 4))[:, :, :3]
            raw_bgr = cv2.resize(raw_bgr, (128, 128))

            # obs["image"] is (4,84,84) already processed
            stacked = obs["image"]          # float32 [0,1]
            state   = obs["state"]          # (6,) float32

            # --- save composite ---
            if total % args.every == 0:
                saved += 1
                comp = build_composite(raw_bgr, stacked, state, action,
                                       float(reward), total)
                img_path = os.path.join(out_dir, f"step_{saved:05d}.png")
                cv2.imwrite(img_path, comp)

                spd, lat, hdg, prv_steer, npc_d, npc_rel = state
                writer.writerow({
                    "step":          total,
                    "speed_ms":      f"{spd:.4f}",
                    "lateral_m":     f"{lat:.4f}",
                    "heading_rad":   f"{hdg:.4f}",
                    "prev_steer":    f"{prv_steer:.4f}",
                    "npc_dist_m":    f"{npc_d:.4f}",
                    "npc_rel_speed": f"{npc_rel:.4f}",
                    "action_steer":  f"{action[0]:.4f}",
                    "action_thr":    f"{action[1]:.4f}",
                    "action_brk":    f"{action[2]:.4f}",
                    "reward":        f"{float(reward):.4f}",
                    "terminated":    int(terminated),
                })
                csv_file.flush()
                print(f"  saved {saved}/{args.steps}  "
                      f"(step={total}  spd={spd:.1f}  lat={lat:.3f}  "
                      f"rew={float(reward):+.2f})")

            if terminated or truncated:
                obs, _ = env.reset()
                if use_vecenv:
                    vec_obs, _ = vec_env.reset()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        csv_file.close()
        env.close()
        print(f"\nDone. {saved} images + state_log.csv saved to:\n  {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect agent-POV samples from CARLA without modifying training code.")
    parser.add_argument("--phase",      required=True,
                        help="Phase name from curriculum.yaml, e.g. p1")
    parser.add_argument("--mode",       choices=["autopilot", "checkpoint"],
                        default="autopilot",
                        help="autopilot = CARLA built-in AI; checkpoint = load PPO model")
    parser.add_argument("--steps",      type=int, default=200,
                        help="Number of composite images to save (default 200)")
    parser.add_argument("--every",      type=int, default=1,
                        help="Save every N environment steps (default 1 = every step)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a .zip checkpoint (default: latest in checkpoints/)")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
