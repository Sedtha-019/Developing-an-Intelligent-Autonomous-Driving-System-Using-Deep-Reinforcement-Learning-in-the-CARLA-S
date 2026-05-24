"""
evaluate.py  —  Deterministic evaluation of the latest PPO checkpoint.

Usage (from project root):
    python training/evaluate.py
    python training/evaluate.py --checkpoint checkpoints/model_1000000.zip
    python training/evaluate.py --episodes 20 --town Town02 --record
"""

import argparse
import random
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

# ── bootstrap sys.path the same way train.py does ──────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.carla_env import CarlaEnv


# ───────────────────────────────────────────────────────────────────────────
# Config helpers
# ───────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the .zip with the highest step number in its name."""
    zips = list(checkpoint_dir.glob("model_*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}. "
            "Run training/train.py first."
        )
    # sort by the integer embedded in the filename: model_500000.zip -> 500000
    zips.sort(key=lambda p: int(p.stem.split("_")[1]))
    return zips[-1]


# ───────────────────────────────────────────────────────────────────────────
# Episode runner
# ───────────────────────────────────────────────────────────────────────────

def run_episodes(
    model: PPO,
    vec_env: VecNormalize,
    n_episodes: int,
    render: bool = False,
) -> list[dict]:
    """
    Roll out `n_episodes` with a deterministic policy.
    Returns a list of per-episode result dicts.

    CarlaEnv.step() returns an empty info dict {}, so termination is inferred
    from the terminated / truncated flags that DummyVecEnv merges into done_arr
    and stores in info_arr under the SB3 keys 'TimeLimit.truncated' and
    'terminal_observation'.  The most reliable approach is to check the raw
    env's own state after the episode ends.
    """
    results = []

    # DummyVecEnv exposes the unwrapped envs at vec_env.venv.envs (through the
    # VecNormalize wrapper) so we can read CarlaEnv attributes directly.
    raw_env = vec_env.venv.envs[0]

    for ep in range(1, n_episodes + 1):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        max_speed = 0.0
        speeds = []
        collided_this_ep = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_arr = vec_env.step(action)

            done = bool(done_arr[0])
            ep_reward += float(reward[0])
            ep_steps += 1

            # Speed from the vehicle directly (m/s planar speed)
            if raw_env.vehicle is not None:
                v = raw_env.vehicle.get_velocity()
                speed = float((v.x**2 + v.y**2) ** 0.5)
                speeds.append(speed)
                if speed > max_speed:
                    max_speed = speed

            # Track collision via CarlaEnv's own flag
            if raw_env.collided:
                collided_this_ep = True

        # Infer termination: SB3 sets 'TimeLimit.truncated'=True in info when
        # the episode ended by truncation (step limit), not by terminated=True.
        truncated = bool((info_arr[0] or {}).get("TimeLimit.truncated", False))
        if collided_this_ep:
            termination = "collision"
        elif truncated:
            termination = "timeout"
        else:
            termination = "off_road"

        result = {
            "episode":       ep,
            "reward":        round(ep_reward, 3),
            "steps":         ep_steps,
            "termination":   termination,
            "collided":      collided_this_ep,
            "max_speed_ms":  round(max_speed, 2),
            "avg_speed_ms":  round(float(np.mean(speeds)), 2) if speeds else 0.0,
        }
        results.append(result)

        status_icon = "✓" if termination == "timeout" else "✗"
        print(
            f"  [{status_icon}] ep {ep:>3} | "
            f"reward {ep_reward:>8.2f} | "
            f"steps {ep_steps:>5} | "
            f"end: {termination}"
        )

    return results


# ───────────────────────────────────────────────────────────────────────────
# Summary printer
# ───────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict], checkpoint_path: Path) -> dict:
    rewards      = [r["reward"]       for r in results]
    steps        = [r["steps"]        for r in results]
    max_speeds   = [r["max_speed_ms"] for r in results]
    avg_speeds   = [r["avg_speed_ms"] for r in results]
    terminations = [r["termination"]  for r in results]

    n = len(results)
    n_timeout   = terminations.count("timeout")
    n_collision = terminations.count("collision")
    n_offroad   = terminations.count("off_road")
    n_collided  = sum(1 for r in results if r.get("collided", False))

    summary = {
        "checkpoint":         str(checkpoint_path),
        "n_episodes":         n,
        "reward_mean":        round(float(np.mean(rewards)),  3),
        "reward_std":         round(float(np.std(rewards)),   3),
        "reward_min":         round(float(np.min(rewards)),   3),
        "reward_max":         round(float(np.max(rewards)),   3),
        "steps_mean":         round(float(np.mean(steps)),    1),
        "avg_speed_mean_ms":  round(float(np.mean(avg_speeds)), 2),
        "max_speed_mean_ms":  round(float(np.mean(max_speeds)), 2),
        "completion_rate_pct": round(100 * n_timeout   / n, 1),
        "collision_rate_pct":  round(100 * n_collision / n, 1),
        "offroad_rate_pct":    round(100 * n_offroad   / n, 1),
    }

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Evaluation summary  ({n} episodes)")
    print(sep)
    print(f"  Checkpoint   : {checkpoint_path.name}")
    print(f"  Reward       : {summary['reward_mean']:>8.2f}  ±{summary['reward_std']:.2f}")
    print(f"               : min {summary['reward_min']:.2f}  /  max {summary['reward_max']:.2f}")
    print(f"  Avg steps    : {summary['steps_mean']:>8.1f}")
    print(f"  Avg speed    : {summary['avg_speed_mean_ms']:>8.2f} m/s")
    print(f"  Completion   : {n_timeout:>3}/{n}  ({summary['completion_rate_pct']}%)")
    print(f"  Collisions   : {n_collision:>3}/{n}  ({summary['collision_rate_pct']}%)")
    print(f"  Off-road     : {n_offroad:>3}/{n}  ({summary['offroad_rate_pct']}%)")
    print(sep)

    return summary


# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CARLA PPO checkpoint.")
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a specific model_<steps>.zip. "
             "Defaults to the latest in checkpoints/.",
    )
    p.add_argument(
        "--vecnorm", type=str, default=None,
        help="Path to vecnormalize.pkl. "
             "Defaults to checkpoints/vecnormalize.pkl.",
    )
    p.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10).",
    )
    p.add_argument(
        "--town", type=str, default=None,
        help="Override the CARLA town (e.g. Town02). "
             "Defaults to the first town in curriculum phase 1.",
    )
    p.add_argument(
        "--weather", type=str, default="ClearNoon",
        help="Weather preset to use during eval (default: ClearNoon).",
    )
    p.add_argument(
        "--traffic", type=int, default=0,
        help="Number of NPC vehicles during eval (default: 0 = clean eval).",
    )
    p.add_argument(
        "--record", action="store_true",
        help="Save per-episode JSON results to logs/eval_<timestamp>.json.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Seed for spawn-point selection. Default (None) uses system "
             "entropy so each episode starts on a random road.",
    )
    p.add_argument(
        "--no-render", dest="render", action="store_false",
        help="Skip any rendering hooks (default: no rendering).",
    )
    p.set_defaults(render=False)
    p.add_argument(
        "--video", action="store_true",
        help="Record an MP4 per episode from chase-camera angles attached to "
             "the ego vehicle. Files land in --video-dir.",
    )
    p.add_argument(
        "--video-angles", type=str, default="chase",
        help="Comma-separated angles to record: chase, front, top "
             "(default: chase).",
    )
    p.add_argument(
        "--video-fps", type=int, default=20,
        help="Output video FPS. Default 20 matches the sim 0.05s dt (1:1 "
             "real-time playback).",
    )
    p.add_argument(
        "--video-dir", type=str, default=None,
        help="Directory for recorded MP4s. Defaults to logs/videos.",
    )
    p.add_argument(
        "--video-res", type=str, default="480x270",
        help="Per-camera recording resolution WxH (default: 480x270).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── paths ───────────────────────────────────────────────────────────────
    checkpoint_dir = ROOT / "checkpoints"
    log_dir        = ROOT / "logs"
    cfg_cur        = load_yaml(ROOT / "configs" / "curriculum.yaml")

    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint
        else find_latest_checkpoint(checkpoint_dir)
    )
    vecnorm_path = (
        Path(args.vecnorm) if args.vecnorm
        else checkpoint_dir / "vecnormalize.pkl"
    )

    if not checkpoint_path.exists():
        sys.exit(f"[error] Checkpoint not found: {checkpoint_path}")
    if not vecnorm_path.exists():
        sys.exit(
            f"[error] vecnormalize.pkl not found: {vecnorm_path}\n"
            "        Without normalisation stats the policy input distribution\n"
            "        will differ from training — evaluation would be meaningless."
        )

    # ── resolve town ────────────────────────────────────────────────────────
    eval_town = args.town or cfg_cur["phases"][0]["towns"][0]

    print(f"\n{'─'*52}")
    print(f"  CARLA PPO — evaluation run")
    print(f"{'─'*52}")
    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"  VecNormalize : {vecnorm_path}")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Town         : {eval_town}")
    print(f"  Weather      : {args.weather}")
    print(f"  NPC traffic  : {args.traffic}")
    print(f"{'─'*52}\n")

    # ── lock curriculum to eval settings for every episode reset ───────────
    # CarlaEnv reads configs itself and uses ScenarioManager for town/weather/
    # traffic. We monkey-patch ScenarioManager.select after construction so it
    # always returns our fixed eval values regardless of global_step.
    eval_traffic = args.traffic

    video_dir = Path(args.video_dir) if args.video_dir else (log_dir / "videos")
    try:
        vw, vh = (int(x) for x in args.video_res.lower().split("x"))
    except ValueError:
        sys.exit(f"[error] --video-res must look like 480x270, got {args.video_res!r}")
    video_angles = tuple(
        a.strip() for a in args.video_angles.split(",") if a.strip()
    )

    if args.video:
        print(f"  Recording    : {','.join(video_angles)} @ {vw}x{vh} {args.video_fps}fps → {video_dir}")

    def make_env():
        env = CarlaEnv(
            global_step=0,
            record_video=args.video,
            video_dir=str(video_dir),
            video_angles=video_angles,
            video_resolution=(vw, vh),
            video_fps=args.video_fps,
        )
        # Override scenario selection so resets don't randomise town/weather/NPC
        env.scenario.select = lambda step: (eval_town, args.weather, eval_traffic)
        return env

    vec_env = DummyVecEnv([make_env])

    # ── load VecNormalize stats (inference mode: do NOT update running stats) ─
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False        # freeze running mean/var
    vec_env.norm_reward = False     # raw rewards for eval readability

    # ── load policy ─────────────────────────────────────────────────────────
    print("Loading policy …")
    model = PPO.load(str(checkpoint_path), env=vec_env)
    print("Policy loaded. Starting evaluation …\n")

    # PPO.load restores the seed saved with the checkpoint, which reseeds
    # Python's `random` module deterministically. CarlaEnv.reset() picks the
    # ego spawn via random.shuffle(spawn_points), so without reseeding here
    # every evaluation run would start on the same road. Reseed with fresh
    # entropy (or args.seed for a reproducible run).
    random.seed(args.seed)

    # ── run ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    results = run_episodes(model, vec_env, args.episodes, render=args.render)
    elapsed = time.time() - t0

    vec_env.close()

    # ── summary ─────────────────────────────────────────────────────────────
    summary = print_summary(results, checkpoint_path)
    summary["eval_time_s"] = round(elapsed, 1)
    summary["town"]        = eval_town
    summary["weather"]     = args.weather
    summary["n_traffic"]   = args.traffic

    # ── save results ────────────────────────────────────────────────────────
    if args.record:
        log_dir.mkdir(exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = log_dir / f"eval_{ts}.json"
        payload   = {"summary": summary, "episodes": results}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Results saved → {out_path}")

    print()


if __name__ == "__main__":
    main()