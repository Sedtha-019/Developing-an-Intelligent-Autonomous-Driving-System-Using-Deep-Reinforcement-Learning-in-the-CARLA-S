# Thesis Project Brief

**Title:** Developing an Intelligent Autonomous Driving System Using Deep Reinforcement Learning in the CARLA Simulator

This document is a structured, factual brief of the project's design, implementation, and rationale, intended as source material for expansion into a full thesis report. It is organized by thesis chapter for direct reuse.

---

## 1. Problem statement

Autonomous driving requires an agent to perceive its surroundings, infer intent of other road users, and continuously produce safe low-level control commands (steering, throttle, brake). Classical modular pipelines (perception → planning → control) decompose this into hand-engineered stages, each with its own assumptions and failure modes. End-to-end deep reinforcement learning (DRL) offers an alternative: learn the perception-to-control mapping directly from interaction.

This project investigates whether a single end-to-end DRL agent — trained on consumer-grade hardware (laptop GPU, 8 GB VRAM) — can learn to drive in the CARLA simulator across multiple towns, weather conditions, and traffic levels using only on-board sensors (a forward-facing RGB camera) and minimal kinematic state.

## 2. Research questions

1. Can a PPO agent trained on a single CARLA instance, on a laptop-class GPU, learn lane-keeping, speed maintenance, and basic collision avoidance from raw camera input?
2. How much shaping must the reward function include before the agent escapes degenerate local optima (e.g., remaining stationary, oscillating the steering wheel)?
3. To what extent does a manually-staged curriculum (single town per phase, single weather, progressively added traffic) improve sample efficiency compared to randomized environment exposure?
4. Which observation features beyond raw pixels (lateral offset, heading deviation, nearest-vehicle distance) most accelerate convergence?

## 3. System overview

The system is a single-agent PPO trainer that drives an ego vehicle in CARLA 0.9.15+, running in synchronous mode at 20 Hz. Episodes are limited to 1000 simulator ticks (50 s of simulated driving). The agent receives a multi-modal observation (image + state vector), produces continuous low-level control, and is trained with shaped reward and curriculum-staged scenarios.

### 3.1. Architecture

- **Algorithm:** Proximal Policy Optimization (PPO) from Stable-Baselines3, multi-input policy
- **Feature extractor:** Nature-style CNN over the image branch + small MLP over the state-vector branch, concatenated and projected to a 256-dim feature
  - CNN: Conv2d(4→32, k=8, s=4) → ReLU → Conv2d(32→64, k=4, s=2) → ReLU → Conv2d(64→64, k=3, s=1) → ReLU → Flatten
  - State MLP: Linear(state_dim → 64) → ReLU
  - Fusion: Linear(cnn_features + 64 → 256) → ReLU
- **Simulator:** CARLA 0.9.15+, synchronous mode, `fixed_delta_seconds = 0.05` (20 Hz), headless rendering
- **Framework:** Stable-Baselines3 (PyTorch), gymnasium

### 3.2. Observation space

A `Dict` observation:

- `image`: 4 × 84 × 84 grayscale frame stack, normalized to [0, 1]
  - Stacked over 4 consecutive timesteps to provide motion cues
  - Grayscale chosen for memory budget; 84×84 follows the Nature-DQN convention
- `state`: 6-dim float vector
  - `[0]` Forward speed (m/s, signed)
  - `[1]` Lateral offset from lane center (m, signed; projected onto lane's right vector)
  - `[2]` Heading deviation from road direction (rad, signed)
  - `[3]` Previous steering command (-1 to 1)
  - `[4]` Distance to nearest leading vehicle within 30m forward cone (m)
  - `[5]` Relative forward velocity of that vehicle (m/s, negative = closing)

The state-vector entries are normalized by `VecNormalize` (running mean/variance) during training. The image branch is not normalized further — only the per-pixel /255 scaling done in preprocessing.

### 3.3. Action space

Continuous, `Box([-1, 0, 0], [1, 1, 1])`:
- `steer` ∈ [-1, 1]
- `throttle` ∈ [0, 1]
- `brake` ∈ [0, 1]

Throttle and brake are decoupled (the agent may command both simultaneously, in which case the vehicle dynamics resolve the net force).

### 3.4. Reward function

Per-step reward is the sum of dense shaping terms and a sparse terminal penalty:

| Term | Formula | Purpose |
|---|---|---|
| Progress | `0.10 · clamp(v_fwd, 0, 15)` | Encourage forward motion |
| Speed tracking | `0.05 · (1 − |v_fwd − 6| / 6)`, or `−0.10` if `v_fwd < 0.5` | Target ~6 m/s, penalize idling |
| Steering magnitude | `−0.05 · |steer| · speed_scale` | Discourage unnecessary steering |
| Steering rate | `−0.05 · (steer − prev_steer)² · speed_scale` | Smooth control |
| Lane deviation | `−0.20 · (lat_dist / 1.75)² · speed_scale` | Stay centered in lane |
| NPC proximity | `−0.20 · (1 − npc_dist / 8)` when within 8m | Avoid tailgating |
| Terminal | `−100` on collision, off-road, or stuck-timeout | Strong "this is bad" signal |

Where `speed_scale = min(v_fwd / 6, 1)` — penalties that depend on driving behavior are zeroed out while the agent is stationary, so it isn't punished for exploring before it has learned to move.

### 3.5. Termination conditions

An episode ends on the first of:

- **Collision** (terminated): collision sensor fires.
- **Off-road** (terminated): vehicle's position has no driving-lane waypoint under it.
- **Stuck-timeout** (terminated): vehicle's |speed| < 0.5 m/s for 200 consecutive ticks (10 s). Without this, an early-policy agent could absorb only minor per-step penalties for the full 1000-step horizon without ever crashing, producing no learning signal.
- **Step limit** (truncated): 1000 simulator ticks reached.

All three terminated paths apply the −100 terminal penalty to provide a consistent failure signal.

## 4. Curriculum design

Training is organized into four sequential phases, each defined by `(town, weather, traffic_count, target_step)` in `configs/curriculum.yaml`:

| Phase | Town | Weather | NPCs | Target step | Purpose |
|---|---|---|---|---|---|
| p1 | Town01 | ClearNoon | 0 | 1.0M | Learn basic driving: lane keeping, steady speed, smooth steering |
| p2 | Town01 | ClearNoon | 10 | 1.7M | Introduce traffic on the familiar map |
| p3 | Town03 | ClearNoon | 0 | 2.5M | Generalize lane-keeping to a more complex map |
| p4 | Town03 | ClearNoon | 10 | 3.5M | Combine new map + traffic |

### 4.1. Why manual per-phase training

The training script (`training/train.py`) is invoked **once per phase** with a `--phase` CLI flag. It auto-resumes from the latest checkpoint, trains until the phase's `target_step` is reached, and exits. The user then closes CARLA, reopens it, and launches the next phase.

This design is dictated by a hardware constraint: on the target laptop (RTX 4060 8 GB), CARLA's `world.load_world()` call leaks memory and the simulator crashes after a small number of repeated invocations. By restricting each training session to a single town, `load_world` is called exactly once per session, eliminating the failure mode.

### 4.2. Why ClearNoon only

The agent's feature extractor is a small CNN (≈100k parameters in the convolutional stack). Adding weather variability multiplies the per-pixel distribution the CNN must model, materially slowing convergence. Fixing weather to `ClearNoon` for the duration of training trades generalization breadth for sample efficiency. Weather randomization would be a natural extension once the agent has converged on the simpler distribution.

### 4.3. Why a stuck-timeout

In early experiments, the initial PPO policy (Gaussian-clipped actions, std ≈ 1) produced approximately zero net force on the ego vehicle (throttle and brake means cancel). The agent remained stationary for full 1000-step episodes, accumulating per-step noise penalties dominated by random steering jitter. With no termination signal, the value function could not learn — explained variance remained at 0 for many iterations. Introducing a 10-second stuck-timeout (terminate + −100 penalty if speed < 0.5 m/s for 200 ticks) provides a clear "do not sit still" signal and ensures episodes do not run their full horizon while doing nothing.

### 4.4. Why filter spawn points

CARLA exposes ~250 spawn points per town. Many are mid-intersection, on tight curves, or facing oncoming lanes — situations where even a perfect policy crashes within seconds. To prevent these "unfair" starts from contaminating the learning signal, the environment filters spawns by:
1. Excluding spawns whose waypoint is in a junction.
2. Excluding spawns whose next 15m of road enters a junction.
3. Excluding spawns whose cumulative road-heading change over 15m exceeds 30°.

The resulting safe-spawn set (typically 30–80% of the original) is cached on first reset. Episodes still encounter junctions and curves during the 50 s of driving — only the start state is restricted to provide a brief "runway."

## 5. Hardware-driven design decisions

| Constraint | Decision | Rationale |
|---|---|---|
| 8 GB VRAM, headless CARLA uses 3–6 GB | Single training environment (`DummyVecEnv` with n=1) | Multiple CARLA instances impossible |
| Limited GPU compute | 84×84 grayscale image | Reduces CNN compute; sufficient for road structure |
| CARLA `load_world` instability | Single-town per training session | Avoids mid-session map reloads |
| Limited sample budget (~3.5M steps total) | Rich state vector + dense reward shaping | Reduces reliance on the CNN to learn perception, planning, and control jointly |
| User wants to pause/resume | Auto-load latest checkpoint + Ctrl+C-safe save | `try/finally` block in `train.py` saves on any exit |

## 6. Engineering implementation

### 6.1. Code structure
```
configs/                Curriculum, environment, PPO hyperparameters (YAML)
env/carla_env.py        Gymnasium environment wrapping CARLA
env/observation.py      Image preprocessing (resize, grayscale, normalize)
env/frame_stack.py      4-frame temporal stacking
env/reward.py           Per-step reward computation
agents/feature_extractor.py    Nature-CNN + state MLP + fusion
training/train.py       Manual per-phase entry point with auto-resume
training/callbacks.py   Periodic checkpoint (50k steps) + VecNormalize save
```

### 6.2. Observation normalization
`VecNormalize` (Stable-Baselines3) maintains running statistics over the `state` vector and rewards only. The image branch is excluded from `VecNormalize` (`norm_obs_keys=["state"]`) — running normalization over pixels would corrupt vision learning by shifting the input distribution after the CNN has already adapted to it.

### 6.3. Checkpointing
- Periodic: every 50,000 environment steps via `CheckpointCallback`
- On exit: `try/finally` block in `train.py` saves on both normal completion and `KeyboardInterrupt`
- Files: `checkpoints/model_<step>.zip` (PPO model + optimizer state) and `checkpoints/vecnormalize.pkl` (running statistics)

### 6.4. Resume semantics
`train.py` scans `checkpoints/` for the highest-step `model_*.zip`, loads PPO + VecNormalize from there, passes `global_step=ckpt_step` to the environment (so internal counters align), and calls `model.learn(total_timesteps=remaining, reset_num_timesteps=False)`. Training resumes seamlessly with consistent TensorBoard step numbers.

## 7. PPO hyperparameters

```yaml
learning_rate: 1e-4
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

`n_steps × n_envs = 2048` per rollout. With `batch_size=256`, that yields 8 minibatches per epoch and 32 gradient updates per rollout. `target_kl=0.02` triggers early stopping of epochs when the policy KL divergence exceeds the threshold, preventing destructive updates.

## 8. Expected results and evaluation

The agent is expected to demonstrate, after the full ~3.5M training steps:

- **Lane keeping:** lateral RMS error < 0.5m on straight road segments
- **Speed maintenance:** mean forward speed within ±1 m/s of the 6 m/s target during cruise
- **Smooth control:** steering rate variance below initial-policy baseline by >50%
- **Collision avoidance:** reaching episode truncation (1000 steps, no crash) in >70% of episodes in p4
- **Cross-town generalization:** comparable lane-keeping performance in Town03 (p3/p4) to Town01 (p1/p2) within ~200k steps of town introduction

Evaluation is performed offline against held-out spawn points and (optionally) against a town not seen during training (e.g., Town02) to measure generalization. Metrics logged via TensorBoard include `ep_rew_mean`, `ep_len_mean`, `explained_variance`, `value_loss`, `approx_kl`, and `clip_fraction`.

## 9. Known limitations and future work

- **No route or goal signal.** The reward shapes lane-following and obstacle avoidance but provides no preference for which way to turn at junctions. The agent's behavior at intersections is arbitrary. A future iteration would add a sparse waypoint-tracking reward or imitation-learning warmup from CARLA's expert autopilot.
- **No pedestrians or cyclists.** Phases 2 and 4 introduce only vehicle traffic. Vulnerable road users would require additional perception and reward terms.
- **Forward-cone-only NPC detection.** The nearest-NPC feature only considers vehicles in front of the ego. Side and rear vehicles are invisible to the state vector and must be inferred from pixels alone.
- **Fixed weather during training.** Only `ClearNoon` is used. Generalization to fog, rain, or night conditions is not measured.
- **Single learning rate, no schedule.** A linear LR decay or KL-adaptive schedule could improve late-training stability.
- **No domain randomization.** Camera intrinsics, position, and noise are fixed. Sim-to-real transfer would require domain randomization or sim-to-real fine-tuning.

## 10. Reproducibility

- Source: this repository (commit `f247ad3` and later)
- CARLA version: 0.9.15
- Python: 3.11
- PyTorch: 2.x (CUDA)
- Stable-Baselines3: 2.x
- Hardware tested on: NVIDIA RTX 4060 Laptop (8 GB VRAM), 16 GB RAM, Windows 11
- Training wall-clock: approximately 3 hours per 1M environment steps at ~93 fps; full curriculum ~10–11 hours

---

## Appendix A — Key references for thesis literature review

- Schulman, J. et al. *Proximal Policy Optimization Algorithms*, 2017.
- Mnih, V. et al. *Human-level control through deep reinforcement learning* (Nature DQN), 2015.
- Dosovitskiy, A. et al. *CARLA: An Open Urban Driving Simulator*, 2017.
- Toromanoff, M., Wirbel, E., Moutarde, F. *End-to-End Model-Free Reinforcement Learning for Urban Driving Using Implicit Affordances* (CVPR 2020).
- Chen, D., Koltun, V., Krähenbühl, P. *Learning by Cheating* (CoRL 2019).
- Bewley, A. et al. *Learning to Drive from Simulation without Real World Labels* (ICRA 2019).
- Kendall, A. et al. *Learning to Drive in a Day* (ICRA 2019).

## Appendix B — Reward function tuning notes

The current reward emerged from observing two failure modes in earlier iterations:

1. **Stationary equilibrium.** Initial reward (`steer_rate_k = 0.5`, no lane penalty) produced a degenerate policy: throttle/brake mean cancelled, car remained stationary, episode ran the full 1000 steps absorbing per-step penalties dominated by random steering jitter (≈ −0.65/step × 1000 = −650 ep_rew). The value function could not predict this noise-dominated signal (explained variance ≈ 0). Resolved by (a) lowering `steer_rate_k` 10×, (b) scaling all steering penalties by `speed_scale` so they vanish when stationary, (c) adding the stuck-timeout termination.

2. **Wobble/zig-zag at speed.** With weakened steering penalties and no lateral signal, the agent learned to drive forward while oscillating the wheel — net per-step reward was still positive because progress dominated. Resolved by adding a quadratic lane-deviation penalty (`−0.20 · (lat_dist / 1.75)²`) that grows quickly near lane edges and rewards staying centered.

These two iterations are documentable as a small ablation study in the thesis.
