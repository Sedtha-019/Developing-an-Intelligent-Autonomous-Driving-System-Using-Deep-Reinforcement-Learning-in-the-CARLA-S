# CARLA RL Autonomous Driving (v3)

End-to-end PPO agent for autonomous driving in CARLA 0.9.15. Learns directly from raw sensor data (camera + kinematic state) using a 4-phase curriculum — from simple lane-keeping on Town01 to navigating traffic on the more complex Town03.

**Algorithm:** PPO (Stable-Baselines3) · **Vision:** 4-frame grayscale stack (84×84) · **Hardware target:** RTX 4060 8 GB laptop, single env

---

## Demo

| Town01 (Phase 1 & 2) | Town03 (Phase 3 & 4) |
|---|---|
| ![Town01 demo](logs/videos/Town01-1.mp4) | ![Town03 demo](logs/videos/town03-1.mp4) |

> Videos recorded at the end of the respective curriculum phase. Agent is running the learned policy, no autopilot.

---

## Architecture

```
Observation (Dict)
  ├── image  (4, 84, 84) float32   ← 4-frame grayscale stack from 128×128 RGB camera
  └── state  (6,)        float32   ← speed, lateral offset, heading dev, prev steer,
                                      nearest NPC dist, nearest NPC rel speed

MultiModalExtractor
  ├── Nature-CNN  (4,84,84) → Conv[32,64,64] → Flatten(3136) ──┐
  └── State MLP   (6,)      → Linear(64)                       ┤→ Linear(3200→256) → ReLU
                                                                ↓
                                              PPO Actor-Critic (256 features)
                                                  ├── Actor   → Action(3,)  [steer, throttle, brake]
                                                  └── Critic  → Value scalar
```

**VecNormalize** is applied to the `state` vector and rewards only — the image is left un-normalized to preserve CNN learning.

---

## Curriculum

| Phase | Town   | NPCs | Steps         | Goal |
|-------|--------|------|---------------|------|
| `p1`  | Town01 | 0    | 0 → 1 M       | Lane-keeping + steady speed |
| `p2`  | Town01 | 10   | 1 M → 1.7 M   | Adapt to traffic on familiar map |
| `p3`  | Town03 | 0    | 1.7 M → 2.5 M | Generalize to harder urban layout |
| `p4`  | Town03 | 10   | 2.5 M → 3.5 M | Combined: new map + traffic |

Each phase runs in a fresh CARLA process (user-controlled restart) to avoid `load_world()` crashes on laptop hardware.

---

## Quick Start

### 1. Install CARLA

Download CARLA 0.9.15 from the [official releases page](https://github.com/carla-simulator/carla/releases). Unzip and install the Python API wheel:

```bash
pip install <CARLA>/PythonAPI/carla/dist/carla-0.9.15-cp311-cp311-win_amd64.whl
```

### 2. Install Python dependencies

```bash
python -m venv carla_rl_env
# Windows
carla_rl_env\Scripts\Activate.ps1
# Linux
source carla_rl_env/bin/activate

pip install -r requirements.txt
```

### 3. Launch CARLA (headless)

```bash
# Windows
CarlaUE4.exe -RenderOffScreen -quality-level=Low

# Linux
./CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

Wait for the server log to settle before starting training.

### 4. Train

```bash
python training/train.py --phase p1     # 0 → 1M steps, Town01, no NPCs
```

When `p1` completes: **close and reopen CARLA**, then:

```bash
python training/train.py --phase p2     # 1M → 1.7M
# close/reopen CARLA
python training/train.py --phase p3     # 1.7M → 2.5M  (Town03)
# close/reopen CARLA
python training/train.py --phase p4     # 2.5M → 3.5M
```

Press `Ctrl+C` at any time to save and exit. Re-running the same phase command resumes automatically from the latest checkpoint.

### 5. Monitor

```bash
tensorboard --logdir logs
# open http://localhost:6006
```

---

## Reward Function

| Term | Formula | Range |
|---|---|---|
| Progress | `0.10 × clamp(speed, 0, 15)` | 0 to +1.5 |
| Speed tracking | `0.05 × (1 − |speed−6| / 6)` or `−0.10` if stationary | −0.10 to +0.05 |
| Steer magnitude | `−0.05 × |steer| × speed_scale` | −0.05 to 0 |
| Steer rate | `−0.05 × (steer − prev_steer)² × speed_scale` | −0.20 to 0 |
| Lane deviation | `−0.20 × (lat_dist / 1.75)² × speed_scale` | −0.80 to 0 |
| NPC proximity | `−0.20 × (1 − dist / 8)` when within 8 m | −0.20 to 0 |
| Terminal | `−100` on collision / off-road / stuck-timeout | −100 or 0 |

`speed_scale = min(speed / 6, 1)` — penalties scale toward zero at low speed so the agent is not punished before it has learned to move.

Set `DEBUG_REWARD=1` to print per-step reward breakdown to stdout.

---

## Episode Termination

| Condition | Type |
|---|---|
| Collision sensor fires | `terminated` |
| No driving-lane waypoint under vehicle (off-road) | `terminated` |
| Speed < 0.5 m/s for 200 consecutive ticks (10 s) | `terminated` |
| 1000 steps reached | `truncated` |

---

## PPO Hyperparameters

```yaml
learning_rate:  0.0001
n_steps:        2048
batch_size:     256
n_epochs:       4
gamma:          0.99
gae_lambda:     0.95
clip_range:     0.2
ent_coef:       0.005
target_kl:      0.02
max_grad_norm:  0.5
checkpoint_freq: 50000
```

---

## Project Layout

```
carla_rl_project_v3/
├── configs/
│   ├── curriculum.yaml        # 4-phase training schedule
│   ├── environment.yaml       # CARLA host/port, sync rate, camera, episode settings
│   └── training.yaml          # PPO hyperparameters
├── env/
│   ├── carla_env.py           # Gymnasium wrapper: step, reset, termination, video
│   ├── reward.py              # Per-step reward computation
│   ├── observation.py         # Image preprocessing (resize → grayscale → normalize)
│   ├── frame_stack.py         # 4-frame temporal stacking
│   └── weather_manager.py     # Weather preset application
├── agents/
│   └── feature_extractor.py   # MultiModalExtractor: CNN + MLP → 256 features
├── training/
│   ├── train.py               # Main entry point: --phase pN, auto-resume
│   ├── callbacks.py           # CheckpointCallback (50k steps + exit save)
│   └── evaluate.py            # Evaluation script
├── tools/
│   └── collect_samples.py     # Step-by-step visualization with state CSV
├── logs/
│   ├── monitor.csv            # Episode reward/length history
│   ├── PPO_0/                 # TensorBoard event files
│   └── videos/
│       ├── Town01-1.mp4       # Demo: Town01 (phases 1–2)
│       └── town03-1.mp4       # Demo: Town03 (phases 3–4)
└── requirements.txt
```

---

## Hardware Requirements

| Component | Recommended |
|---|---|
| GPU | RTX 4060 8 GB (or equivalent) |
| RAM | 16 GB |
| Disk | SSD, ~10 GB free |
| OS | Windows 10/11 or Linux |

VRAM budget breakdown on 8 GB:
- Headless CARLA (Town01, no NPCs, Low quality): 3–4 GB
- Headless CARLA (Town03 + 10 NPCs, Low quality): 4.5–5.5 GB
- PPO model + rollout buffer: ~0.5 GB

**Single environment only.** A second `CarlaEnv` would require a second CARLA instance and exceeds the VRAM budget.

---

## Common Issues

| Error | Fix |
|---|---|
| `time-out of 20000ms` | CARLA isn't running or still loading. Wait for server log to settle. |
| `Camera produced no image after 40 ticks` | Restart CARLA with `-RenderOffScreen -quality-level=Low`. |
| Training hangs at `world.tick()` | Another client is connected. Only one trainer per server. |
| GPU OOM in p3/p4 | Lower NPC count in `curriculum.yaml`; confirm `-quality-level=Low`. |
| `Phase 'pX' not found` | Phase name in `--phase` doesn't match any `name:` entry in `curriculum.yaml`. |
| `ModuleNotFoundError: env` | Run from project root: `python training/train.py`, not `cd training && python train.py`. |
| Stale checkpoint after changing state vector size | Delete `checkpoints/` and retrain from p1. Old zip files are incompatible with new input shape. |

---

## License

This project is for academic/research purposes. CARLA simulator is subject to its own [license](https://carla.org/).
