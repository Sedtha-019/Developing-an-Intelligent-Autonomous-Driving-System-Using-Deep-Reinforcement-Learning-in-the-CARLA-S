# CARLA Reinforcement Learning Project (v3)

End-to-end PPO training for a CARLA self-driving agent on a single laptop GPU. Designed for manual, per-phase curriculum training where you (the user) restart CARLA between phases to avoid mid-session `load_world()` crashes.

- Algorithm: PPO (Stable-Baselines3) with multi-input policy (image + state vector)
- Vision: 4-frame grayscale stack at 84x84, fed to a Nature-CNN feature extractor
- State vector: 6-dim (speed, lateral offset, heading dev, prev steer, nearest NPC dist, nearest NPC rel speed)
- Simulator: CARLA 0.9.15+ in synchronous mode, single training environment
- Curriculum: one town per phase, ClearNoon weather, NPC count ramped progressively
- Checkpointing: every 50k steps + on every clean exit (including Ctrl+C)

---

## 1. Project layout

```
carla_rl_project_v3/
|
+-- configs/
|   +-- training.yaml         # PPO hyperparameters
|   +-- environment.yaml      # CARLA host/port, sync settings, camera, episode length, stuck-timeout
|   +-- curriculum.yaml       # Named phases: town + weather + traffic + target_step
|
+-- env/
|   +-- carla_env.py          # gymnasium env: scene metrics, termination, actor cleanup
|   +-- observation.py        # image preprocessing
|   +-- frame_stack.py        # 4-frame stack
|   +-- reward.py             # progress + speed + lane + steer + NPC proximity + terminal
|   +-- scenario_manager.py   # (legacy) step-based phase picker, not used by train.py
|   +-- weather_manager.py    # applies weather preset
|
+-- agents/
|   +-- feature_extractor.py  # Nature-CNN + state MLP fusion -> 256 features
|
+-- training/
|   +-- train.py              # Manual per-phase entry point: `--phase p1` etc.
|   +-- callbacks.py          # 50k-step checkpoint + VecNormalize save
|
+-- checkpoints/              # model_<steps>.zip + vecnormalize.pkl (latest wins)
+-- logs/                     # TensorBoard logs + monitor.csv
+-- requirements.txt
+-- README.md
```

---

## 2. Hardware target

This project is sized for a laptop-class GPU. **The single biggest constraint is CARLA's VRAM usage**, not the policy network.

| Component | Recommended |
|---|---|
| GPU | RTX 4060 (8 GB) or better |
| RAM | 16 GB |
| Disk | SSD, ~30 GB free |
| OS | Windows 10/11 or Linux |

VRAM budget on an 8 GB card:
- Headless CARLA (Town01, no NPCs, low quality): ~3-4 GB
- Headless CARLA (Town03 + 10 NPCs): ~4.5-5.5 GB
- PyTorch model + PPO rollout buffer: ~0.5 GB
- Leave 1-1.5 GB headroom

**Single training env only.** A second `CarlaEnv` would need another CARLA instance and there is no VRAM for it.

---

## 3. Installation

### 3.1. CARLA simulator
Download CARLA 0.9.15 (or later) from the official releases page. Unzip and verify you can run `CarlaUE4.exe` (Windows) or `./CarlaUE4.sh` (Linux).

### 3.2. Python environment
```
python -m venv carla_rl_env
```

Activate it.

Windows (PowerShell):
```
carla_rl_env\Scripts\Activate.ps1
```

Linux / macOS:
```
source carla_rl_env/bin/activate
```

### 3.3. Install dependencies
```
pip install -r requirements.txt
```

Also install the CARLA Python API wheel that ships inside the CARLA install at `PythonAPI/carla/dist/`. Match your Python version:
```
pip install <CARLA>/PythonAPI/carla/dist/carla-0.9.15-cp311-cp311-win_amd64.whl
```

---

## 4. Start CARLA (headless)

CARLA must be running BEFORE you start training. Launch it in headless mode so the GPU is reserved for your policy.

Windows:
```
CarlaUE4.exe -RenderOffScreen -quality-level=Low
```

Linux:
```
./CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

The server listens on `localhost:2000` by default. Wait for the CARLA log to settle before launching training.

If your machine has multiple GPUs, pin CARLA to a specific one with `-graphicsadapter=N`.

---

## 5. Training workflow (manual per-phase)

All commands are run from the project root: `D:\RL\carla_rl_project_v3`.

### 5.1. The phases

Defined in `configs/curriculum.yaml`. Each phase is one town, ClearNoon, single NPC count, with an absolute `target_step` at which the phase is considered complete.

| Phase | Town    | NPCs | Target step | Purpose |
|-------|---------|------|-------------|---------|
| `p1`  | Town01  | 0    | 1,000,000   | Learn to drive: lane keeping + steady speed |
| `p2`  | Town01  | 10   | 1,700,000   | Add traffic on the familiar map |
| `p3`  | Town03  | 0    | 2,500,000   | Generalize to a harder map, no traffic |
| `p4`  | Town03  | 10   | 3,500,000   | Combine new map + traffic |

You can freely edit `target_step` values; the script trains until whatever target is set.

### 5.2. Run a phase

```
python training/train.py --phase p1
```

What happens:

1. Reads `configs/curriculum.yaml`, finds the `p1` entry, locks the env to Town01 / ClearNoon / 0 NPCs.
2. Scans `checkpoints/` for the highest `model_<step>.zip`:
   - **Found** -> loads PPO + `vecnormalize.pkl`, starts `CarlaEnv` at that `global_step`.
   - **Not found** -> creates fresh PPO + VecNormalize.
3. Computes `remaining = target_step - current_step`. If `<= 0`, prints a message and exits (nothing to do).
4. Trains. Saves a checkpoint every 50k steps via `CheckpointCallback`.
5. On reaching `target_step` -> saves & exits.
6. On `Ctrl+C` -> caught in `try/finally`, saves & exits cleanly.

### 5.3. Stopping mid-phase
Press `Ctrl+C`. The `finally` block saves the model and VecNormalize stats to `checkpoints/`. Re-running `python training/train.py --phase p1` resumes from exactly where you stopped.

### 5.4. Moving to the next phase

When `p1` finishes:

1. Close CARLA.
2. Re-launch CARLA (loads the new town fresh -> no `load_world()` crash).
3. Run `python training/train.py --phase p2`.

The script automatically picks up the latest checkpoint from `p1` and continues. Same pattern for `p3` and `p4`.

### 5.5. Complete example

```
# Open CARLA, then:
python training/train.py --phase p1     # trains to 1.0M, saves, exits
# Close CARLA, reopen CARLA, then:
python training/train.py --phase p2     # trains 1.0M -> 1.7M
# Close CARLA, reopen CARLA, then:
python training/train.py --phase p3     # trains 1.7M -> 2.5M (Town03)
# Close CARLA, reopen CARLA, then:
python training/train.py --phase p4     # trains 2.5M -> 3.5M
```

Total wall-clock at ~93 fps: roughly **10-11 hours** of pure training across all four phases.

---

## 6. Monitor in TensorBoard

In a separate terminal:
```
tensorboard --logdir logs
```

Open `http://localhost:6006`. Each resume creates a new `PPO_N` run; the global step on the x-axis is correct because `reset_num_timesteps=False`.

Key metrics to watch:

| Metric | What healthy looks like |
|---|---|
| `rollout/ep_rew_mean` | Trending up, crosses 0 within ~100k steps in p1 |
| `rollout/ep_len_mean` | Rising over time (less stuck-timeouts and crashes) |
| `train/explained_variance` | > 0.1 within ~50k steps, climbing toward 0.5+ |
| `train/approx_kl` | Stays below ~0.03 |
| `train/clip_fraction` | Generally 0.05 - 0.25 |
| `train/value_loss` | Decreasing trend |

**Red flags:**
- `explained_variance` stuck at ~0 after 100k steps -> reward signal is degenerate.
- `ep_len_mean` pinned at the max (1000) -> agent is stationary, stuck-timeout not firing, or terminations not wired.
- `ep_rew_mean` falling -> reward shaping is misaligned.

---

## 7. Configuration

### 7.1. `configs/training.yaml`
```yaml
total_steps: 2000000        # informational only; per-phase target_step is what matters
checkpoint_freq: 50000      # save every 50k env steps

ppo:
  learning_rate: 0.0001
  n_steps: 2048
  batch_size: 256
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.005
  target_kl: 0.02
  max_grad_norm: 0.5
```

### 7.2. `configs/environment.yaml`
```yaml
host: localhost
port: 2000

fixed_delta_seconds: 0.05    # 20 Hz sim, sync mode
max_episode_steps: 1000      # truncation length

stuck_speed_threshold: 0.5   # m/s
stuck_timeout_steps: 200     # 10s @ 20Hz - terminate if no movement for this long

camera:
  width: 128
  height: 128
  fov: 90
```

### 7.3. `configs/curriculum.yaml`
```yaml
phases:
  - name: p1
    town: Town01
    weather: ClearNoon
    traffic: 0
    target_step: 1000000

  - name: p2
    town: Town01
    weather: ClearNoon
    traffic: 10
    target_step: 1700000

  - name: p3
    town: Town03
    weather: ClearNoon
    traffic: 0
    target_step: 2500000

  - name: p4
    town: Town03
    weather: ClearNoon
    traffic: 10
    target_step: 3500000
```

**Constraints (important):**
- One town per phase, one weather per phase.
- All phases use `ClearNoon` (the CNN is small; adding weather variability would significantly slow learning).
- The training script picks the phase by name (`--phase pN`), not by step count.

---

## 8. Observation and action spaces

**Observation (Dict):**

- `image`: `Box(0, 1, shape=(4, 84, 84), float32)` - 4-frame grayscale stack
- `state`: `Box(-inf, +inf, shape=(6,), float32)`
  - `[0]` `forward_speed` (m/s, signed)
  - `[1]` `lateral_offset` (m, signed; left/right of lane center via lane right vector)
  - `[2]` `heading_deviation` (rad, signed; angle between vehicle forward and road forward)
  - `[3]` `prev_steer` (-1 to 1, the last steering command applied)
  - `[4]` `nearest_npc_dist` (m, in forward cone <=30m and <=3m lateral; defaults to 30 if no NPC ahead)
  - `[5]` `nearest_npc_rel_speed` (m/s; negative = closing)

**VecNormalize:** normalizes the `state` vector and rewards. The image is left untouched (running statistics on pixels would corrupt vision learning).

**Action (Box, shape=(3,)):**

- `steer` in `[-1, 1]`
- `throttle` in `[0, 1]`
- `brake` in `[0, 1]`

---

## 9. Reward function

Defined in `env/reward.py`. Per-step total is the sum of:

| Term | Formula | Range |
|---|---|---|
| Progress | `0.10 * clamp(forward_speed, 0, 15)` | 0 to +1.5 |
| Speed tracking | `0.05 * (1 - |speed - 6| / 6)` (or `-0.10` if stationary) | -0.10 to +0.05 |
| Steer | `-0.05 * |steer| * speed_scale` | -0.05 to 0 |
| Delta steer | `-0.05 * (steer - prev_steer)^2 * speed_scale` | -0.20 to 0 |
| Lane deviation | `-0.20 * (lat_dist / 1.75)^2 * speed_scale` | -0.80 to 0 |
| NPC proximity | `-0.20 * (1 - npc_dist / 8)` when within 8m | -0.20 to 0 |
| Terminal | `-100` on collision / off-road / stuck-timeout | -100 or 0 |

`speed_scale = min(speed / 6, 1)` — penalties scale to zero when the car isn't moving, so the agent isn't punished for exploring before it has learned to drive.

Set `DEBUG_REWARD=1` env var to see per-step breakdown printed to stdout.

---

## 10. Episode termination

An episode ends on the first of:

- **Collision** (`terminated`): collision sensor fires.
- **Off-road** (`terminated`): the vehicle's location has no driving-lane waypoint under it.
- **Stuck-timeout** (`terminated`): vehicle's speed has been below 0.5 m/s for `stuck_timeout_steps` consecutive ticks (10s by default).
- **Step limit** (`truncated`): `max_episode_steps` reached (1000).

All three `terminated` paths trigger the -100 terminal penalty so the agent gets a clear "this is bad" signal.

On every reset, all previously spawned actors (ego vehicle, camera, collision sensor, NPCs) are batch-destroyed. The world is only reloaded when the curriculum picks a different town — and since each phase is single-town, this never happens within a training session.

---

## 11. Safe spawn-point filtering

CARLA exposes hundreds of spawn points per town. Many are mid-intersection, on tight curves, or facing oncoming lanes — "unfair" starts where even a perfect policy would crash within a few seconds. To stop these starts from drowning the learning signal in noise, `CarlaEnv` filters them on the first reset:

- Reject any spawn whose waypoint is **in a junction** (`wp.is_junction`)
- Walk forward 15m in 3m steps from the spawn; reject if **any step enters a junction**
- Reject if **cumulative heading change over 15m exceeds 30°** (excludes spawns on sharp curves)

The resulting safe-spawn indices are cached on the env instance (single-town per phase means one-time cost). NPC vehicles still spawn at *any* of the remaining spawn points — only the ego car is restricted.

You'll see one of these printed on the first reset of a session:
```
[CarlaEnv] 78/254 safe spawn points selected for ego
```

This does **not** restrict the agent to straight-road driving. Each episode is ~300m of travel — the agent leaves its "safe runway" within the first 5 seconds and encounters curves, junctions, and traffic for the remaining 45 seconds.

---

---

## 12. Expected learning curve

Rough estimates assuming `lr=1e-4` and the current reward/state setup:

| Phase | Steps required | Expected behavior at end |
|---|---|---|
| p1 (Town01, no NPCs) | ~1M | Drives well on straight roads, handles gentle curves, stays centered |
| p2 (Town01, 10 NPCs) | +0.7M | Avoids leading vehicles, slows for proximity, doesn't tailgate |
| p3 (Town03, no NPCs) | +0.8M | Generalizes to harder roads; expect a brief reward dip on town change as VecNormalize stats re-adapt |
| p4 (Town03, 10 NPCs) | +1.0M | Combined competence |

Numbers are typical, not guaranteed. Reward shaping and CARLA build version both shift the curve.

---

## 13. Common mistakes

- **Starting `train.py` before CARLA is up.** The client times out after 20s.
- **Switching towns mid-session.** Don't manually edit `curriculum.yaml` to mix towns in one phase. CARLA's repeated `load_world()` will crash on a laptop GPU.
- **Loading a stale checkpoint after changing the state vector.** If you modify `vector_state`'s dimension count, old `model_*.zip` files are incompatible with the new policy input size. Delete old checkpoints and start from p1.
- **Deleting `vecnormalize.pkl` while keeping `model_*.zip`.** Resume cannot recover observation/reward stats. Either keep both or delete both.
- **Running multiple training scripts against the same CARLA server.** The world becomes inconsistent.
- **Launching CARLA without `-RenderOffScreen` on a laptop.** Unreal will fight your policy for the GPU and tank fps.

---

## 14. Troubleshooting

**`ModuleNotFoundError: env`**
Run from the project root: `python training/train.py --phase p1` (the script bootstraps `sys.path`). Don't `cd training` first.

**`RuntimeError: Camera produced no image after 40 ticks`**
CARLA isn't in sync mode, the camera failed to attach, or the server is overloaded. Restart CARLA with `-RenderOffScreen -quality-level=Low`.

**Training hangs at `world.tick()`**
Another client is connected and also driving the world. Make sure only one trainer talks to the server.

**`time-out of 20000ms while waiting for the simulator`**
CARLA isn't running on `localhost:2000`, or it's still loading the map. Wait for CARLA logs to settle.

**`Phase 'pX' not found`**
Phase name in `--phase` doesn't match any `name:` entry in `configs/curriculum.yaml`. Check spelling.

**`Latest checkpoint is at step N, already past target M`**
The latest saved step is past this phase's `target_step`. Either move to the next phase or raise the target.

**Sudden ep_rew_mean drop after switching from p2 to p3**
Expected. VecNormalize reward stats are stale on the new town; they re-adapt within ~10-20k steps.

**GPU OOM during p3 or p4**
Town03 + 10 NPCs is more VRAM-hungry than Town01. Re-launch CARLA with `-quality-level=Low` (mandatory) and lower the NPC count in `curriculum.yaml` if needed. Monitor with `nvidia-smi -l 2`.

---

## 15. Stopping and clean shutdown

Press `Ctrl+C`. The script:

1. Catches the `KeyboardInterrupt`.
2. Saves the current `model_<step>.zip` and `vecnormalize.pkl`.
3. Calls the env's `close()`, which destroys all spawned actors and disables sync mode on the world and traffic manager.

Leave CARLA running if you want to resume the same phase. Restart CARLA only when moving to a new phase (different town).

---

## 16. Files NOT used by the current workflow

These files exist in the tree but are not part of the active training flow. Don't delete them blindly — some may still be useful for evaluation.

- `training/resume.py` — superseded by auto-resume in `train.py`.
- `training/train_phase2.py` — superseded by `--phase` flag in `train.py`.
- `env/scenario_manager.py` — only used if `CarlaEnv` is constructed without explicit `town`. Useful for evaluation scripts.
- `training/evaluate.py` — may need state-vector updates before it works with the current 6-dim state.
