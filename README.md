# CARLA Reinforcement Learning Project (v3)

End-to-end PPO training for a CARLA self-driving agent on a single laptop GPU.

- Algorithm: PPO (Stable-Baselines3) with multi-input policy (image + speed)
- Vision: 4-frame grayscale stack at 84x84, fed to a Nature-CNN feature extractor
- Simulator: CARLA 0.9.15+ in synchronous mode, single training environment
- Curriculum: town / weather / NPC traffic switching across phases up to 5M steps
- Checkpointing: every 500k steps, with VecNormalize statistics saved alongside

---

## 1. Project layout

```
carla_rl_project_v3/
|
+-- configs/
|   +-- training.yaml         # PPO + total steps + checkpoint frequency
|   +-- environment.yaml      # CARLA host/port, sync settings, camera, episode length
|   +-- curriculum.yaml       # Town / weather / traffic per phase
|
+-- env/
|   +-- carla_env.py          # gymnasium env: sync ticks, termination, actor cleanup
|   +-- observation.py        # image preprocessing + speed vector
|   +-- frame_stack.py        # 4-frame stack
|   +-- reward.py             # speed reward + collision penalty
|   +-- scenario_manager.py   # picks town/weather/traffic per phase
|   +-- weather_manager.py    # applies weather preset
|
+-- agents/
|   +-- feature_extractor.py  # Nature-CNN + speed MLP fusion -> 256 features
|
+-- training/
|   +-- train.py              # First-time training entry point
|   +-- resume.py             # Resume from latest checkpoint
|   +-- callbacks.py          # Periodic checkpoint + VecNormalize save
|
+-- checkpoints/              # model_<steps>.zip + vecnormalize.pkl
+-- logs/                     # TensorBoard logs
+-- requirements.txt
+-- README.md
```

---

## 2. Hardware target

This project is sized for a laptop class GPU.

| Component            | Recommended           |
|----------------------|-----------------------|
| GPU                  | RTX 4060 (8 GB) or better |
| RAM                  | 16 GB                 |
| Disk                 | SSD, ~30 GB free      |
| OS                   | Windows 10/11 or Linux |

CARLA itself is CPU-bound; the policy network is what uses the GPU.

---

## 3. Installation

### 3.1. CARLA simulator

Download CARLA 0.9.15 (or later) from the official releases page.

Unzip and verify you can run `CarlaUE4.exe` (Windows) or `./CarlaUE4.sh` (Linux).

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

You also need the CARLA Python API (`carla` package) matching your CARLA build. It usually ships inside the CARLA install at `PythonAPI/carla/dist/`. Install the wheel that matches your Python version, e.g.:

```
pip install <CARLA>/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-win_amd64.whl
```

---

## 4. Start CARLA (headless)

CARLA must be running BEFORE training is started. Always launch it in headless mode to keep GPU load on the policy, not on Unreal rendering.

Windows:
```
CarlaUE4.exe -RenderOffScreen -quality-level=Low
```

Linux:
```
./CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

No window is shown. The server listens on `localhost:2000` by default.

If your machine has multiple GPUs, you can pin CARLA to a specific one with `-graphicsadapter=N`.

---

## 5. Training

All commands below are run from the project root: `D:\RL\carla_rl_project_v3`.

### 5.1. First-time training

```
python training/train.py
```

What happens:

1. Reads `configs/training.yaml` (PPO hyperparameters + total step budget).
2. Reads `configs/environment.yaml` (sync delta, camera resolution, max episode steps).
3. Reads `configs/curriculum.yaml` (town / weather / traffic schedule).
4. Connects to CARLA at `localhost:2000` and switches the world into synchronous mode at `fixed_delta_seconds = 0.05` (20 Hz).
5. Spawns ego vehicle, RGB camera, and collision sensor; spawns NPC traffic per the current curriculum phase.
6. Trains PPO. Checkpoints are saved every `checkpoint_freq` steps (default 500k).

The script returns when `total_steps` (default 5,000,000) is reached.

### 5.2. Monitor in TensorBoard

In a separate terminal:

```
tensorboard --logdir logs
```

Open `http://localhost:6006`. or python -m tensorboard.main --logdir logs Watch:

- `rollout/ep_rew_mean` and `rollout/ep_len_mean` (learning signal)
- `train/loss`, `train/policy_gradient_loss`, `train/value_loss` (stability)
- `train/clip_fraction`, `train/approx_kl` (PPO health, KL should stay <~0.03)

### 5.3. Stop training safely

Press `Ctrl + C` in the training terminal. The latest checkpoint on disk is intact; in-progress rollout is discarded.

### 5.4. Resume training

```
python training/resume.py
```

This:

- Loads `checkpoints/latest` (the PPO model).
- Loads `checkpoints/vecnormalize.pkl` (observation/reward statistics).
- Restores `num_timesteps` so curriculum picks up at the right phase.
- Reattaches the checkpoint callback (saves continue).
- Continues writing TensorBoard logs to `logs/`.
- Trains until `total_steps` is reached.

Never load `checkpoints/latest.zip` without `vecnormalize.pkl` -- the agent's input distribution would shift and effectively erase what it has learned.

---

## 6. Configuration

### 6.1. `configs/training.yaml`

```yaml
total_steps: 5000000        # full training budget
checkpoint_freq: 500000     # save every 500k env steps

ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
```

### 6.2. `configs/environment.yaml`

```yaml
host: localhost
port: 2000

fixed_delta_seconds: 0.05    # 20 Hz sim, sync mode
max_episode_steps: 1000      # truncation length

camera:
  width: 128
  height: 128
  fov: 90
```

### 6.3. `configs/curriculum.yaml`

Phases are matched by `step <= until_step`. Edit between runs as you see fit.

```yaml
phases:
  - until_step: 1000000
    towns:   ["Town01"]
    weather: ["ClearNoon"]
    traffic: 0

  - until_step: 3000000
    towns:   ["Town01", "Town03"]
    weather: ["ClearNoon", "CloudyNoon", "WetNoon"]
    traffic: 20

  - until_step: 5000000
    towns:   ["Town01", "Town02", "Town03", "Town05"]
    weather: ["ClearNoon", "CloudyNoon", "WetNoon", "SoftRainNoon"]
    traffic: 50
```

Town / weather / traffic are re-sampled at each episode reset, never mid-rollout.

---

## 7. Episode termination

Episodes end on the first of:

- **Collision** (terminated): collision sensor fires.
- **Off-road** (terminated): the vehicle's location has no driving-lane waypoint under it.
- **Step limit** (truncated): `max_episode_steps` reached.

On every reset, all previously spawned actors (ego vehicle, camera, collision sensor, NPCs) are batch-destroyed before new ones are created. The world is only reloaded when the curriculum picks a different town than the previous episode.

---

## 8. Observation and action spaces

**Observation (Dict):**

- `image`: `Box(0, 1, shape=(4, 84, 84), float32)` -- 4-frame grayscale stack
- `state`: `Box(-inf, +inf, shape=(1,), float32)` -- planar speed in m/s

**VecNormalize:** normalizes `state` and rewards only. The image is left untouched (running statistics on pixels would corrupt vision learning).

**Action (Box, shape=(3,)):**

- `steer` in `[-1, 1]`
- `throttle` in `[0, 1]`
- `brake` in `[0, 1]`

---

## 9. Expected progression

| Phase          | Steps      | Capability               |
|----------------|------------|--------------------------|
| Early          | 0 - 500k   | Random / jerky driving   |
| Learning       | 500k - 1.5M| Lane following emerging  |
| Stable         | 1.5M - 3M  | Smooth steering, traffic-aware |
| Generalization | 3M - 5M    | Multi-town, multi-weather |

Numbers are typical, not guaranteed; reward shaping and CARLA build version both shift the curve.

---

## 10. Common mistakes

- Starting `train.py` before CARLA is up. The client times out after 20 s.
- Deleting `vecnormalize.pkl`. Resume cannot recover observation stats; re-train from scratch.
- Editing the observation space (image size, channels, state dims) and resuming an old checkpoint. Shape mismatch -- start fresh.
- Running multiple training scripts against the same CARLA server. The world becomes inconsistent.
- Launching CARLA without `-RenderOffScreen` on a laptop. Unreal will fight your policy for the GPU.

---

## 11. Troubleshooting

**`ModuleNotFoundError: env`**
Run from the project root: `python training/train.py` (the script bootstraps `sys.path`). Do not `cd training` first.

**`RuntimeError: Camera produced no image after 40 ticks`**
CARLA is not in synchronous mode, the camera failed to attach, or the server is overloaded. Restart CARLA with `-RenderOffScreen -quality-level=Low` and try again.

**Training hangs at `world.tick()`**
Another client is connected and also driving the world. Make sure only one trainer talks to the server.

**`time-out of 20000ms while waiting for the simulator`**
CARLA is not running on `localhost:2000`, or it is still loading the map. Wait for the CARLA logs to settle, then start training.

**GPU OOM**
Lower `camera.width/height` in `environment.yaml` (the policy CNN still receives 84x84 grayscale, but the CARLA -> Python transfer becomes cheaper). Or lower PPO `batch_size`.

---

## 12. Stopping and clean shutdown

Press `Ctrl + C`. The env's `close()` will:

- Destroy all spawned actors.
- Disable synchronous mode on the world and traffic manager.

Leave CARLA running if you intend to resume; restart it only if it has accumulated stale state.

---

## 13. Next experiments (after 5M steps)

- Evaluation script with deterministic policy on held-out towns
- Richer reward (lane-center deviation, jerk penalty, target speed tracking)
- Lane-invasion sensor as soft-termination signal
- More complex traffic and pedestrian scenarios
- Domain randomization (camera noise, sun position, fog)
- Imitation pretraining from a CARLA expert agent

===============================================================================================================
===============================================================================================================
===============================================================================================================

# more episodes for a reliable average
python training/evaluate.py --episodes 20

# test on a town the agent wasn't trained on yet (generalization test)
python training/evaluate.py --town Town02 --episodes 15

# add traffic to stress-test the agent
python training/evaluate.py --traffic 20 --episodes 10

# test a specific checkpoint, not the latest
python training/evaluate.py --checkpoint checkpoints/model_500000.zip

# save results to logs/eval_<timestamp>.json for later comparison
python training/evaluate.py --episodes 20 --record

===============================================================================================================
===============================================================================================================
===============================================================================================================
---

## 14. Phase 2 training (after Phase 1 completes at 1M)

Phase 1 (`training/train.py` / `training/resume.py`) runs Town01-only up to `total_steps`
in `configs/training.yaml`. When Phase 1 is finished, you have:

- `checkpoints/phase1_final.zip` -- the locked Phase 1 policy
- `checkpoints/vecnormalize_phase1.pkl` -- matching observation/reward stats

**Do not overwrite these.** Phase 2 reads from them but writes to a separate directory.

### 14.1. Why Phase 2 is split off

The original `configs/curriculum.yaml` mixes Town01 and Town03 in Phase 2. With random
town selection every reset, CARLA calls `world.load_world(...)` on roughly every other
episode. After ~1M cumulative steps, the UE4 server's memory leak from repeated map
reloads brings the simulator down. Symptoms:

```
INFO: streaming client: connection failed: No connection could be made...
```

The Phase 2 setup avoids this by running **one town per training run**. You train
Town01 to convergence, then switch to Town03 in a new run that resumes from the
Town01 checkpoint.

### 14.2. Phase 2 files

| File                                       | Purpose                                         |
|--------------------------------------------|-------------------------------------------------|
| `configs/training_phase2.yaml`             | Phase 2 hyperparams (lower LR for fine-tuning)  |
| `configs/curriculum_phase2_town01.yaml`    | Town01 only, weather variety, 20 NPCs           |
| `configs/curriculum_phase2_town03.yaml`    | Town03 only, weather variety, 20 NPCs           |
| `training/train_phase2.py`                 | Phase 2 entry point (seeds + resumes)           |
| `checkpoints_phase2/`                      | Phase 2 checkpoints (auto-created)              |
| `logs_phase2/`                             | Phase 2 TensorBoard logs (auto-created)         |

### 14.3. Running Phase 2

**First run -- Town01 with traffic and weather variety:**

```
python training/train_phase2.py --curriculum configs/curriculum_phase2_town01.yaml
```

The script:

1. Detects no checkpoint in `checkpoints_phase2/`, so seeds from
   `checkpoints/phase1_final.zip` + `vecnormalize_phase1.pkl`.
2. Continues `num_timesteps` from 1,000,000 onward.
3. Saves a checkpoint every 25,000 steps to `checkpoints_phase2/model_<steps>.zip`.
4. Writes TensorBoard logs to `logs_phase2/`.

Let it run until you're happy with the Town01 performance (e.g. step ~2,000,000), then
`Ctrl + C`.

**Second run -- switch to Town03 (continues from latest Phase 2 checkpoint):**

```
python training/train_phase2.py --curriculum configs/curriculum_phase2_town03.yaml
```

This automatically resumes from the highest `model_*.zip` in `checkpoints_phase2/`, so
the policy keeps its Phase 2 progress and only the town/weather/traffic changes.

### 14.4. TensorBoard for Phase 2

In a separate terminal:

```
tensorboard --logdir logs_phase2
```

Compare to Phase 1 (`tensorboard --logdir logs`) side by side by running two
TensorBoard instances on different ports, or run `tensorboard --logdir_spec=p1:logs,p2:logs_phase2`.

### 14.5. Phase 2 stopping rules

- `ep_rew_mean` should initially **dip** when you swap to Town03 (new map, unseen
  geometry). It should recover within ~100-200k steps if the policy generalizes.
- If `value_loss` spikes and stays high beyond ~50k steps after the town switch,
  the new town is too different; consider lowering `learning_rate` to 2e-5 or
  warming up with more Town01 first.
- If `clip_fraction` climbs above ~0.25 sustained, that's the policy thrashing on
  the new objective -- raise `n_steps` (smoother gradient) or lower the LR.

### 14.6. Phase 2 safety rules

- **Never** delete `checkpoints/phase1_final.zip` or `checkpoints/vecnormalize_phase1.pkl`.
  Phase 2 only seeds from them on the first run, but they are your only fallback
  if Phase 2 diverges.
- **Never** train Phase 2 with a curriculum that lists more than one town. The whole
  reason for splitting Phase 2 is to avoid mid-run `load_world()` thrashing.
- If CARLA dies during a Phase 2 run, simply restart CARLA and re-run the same
  `train_phase2.py` command -- it auto-resumes from the latest `checkpoints_phase2/`
  snapshot.
- Lower the `checkpoint_freq` in `configs/training_phase2.yaml` further (e.g. to
  10,000) if Phase 2 is on a town that crashes CARLA more often -- shorter checkpoint
  intervals = less lost work per crash.