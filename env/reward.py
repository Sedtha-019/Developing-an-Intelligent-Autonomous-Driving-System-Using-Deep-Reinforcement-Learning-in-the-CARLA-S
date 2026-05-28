import os

TARGET_SPEED_MS = 6.0
TERMINAL_PENALTY = 100.0

PROGRESS_K = 0.10
SPEED_K = 0.05
STATIC_PENALTY = 0.10
STATIC_SPEED_THRESHOLD = 0.5

STEER_K = 0.05
STEER_RATE_K = 0.05

LANE_HALF_WIDTH_M = 1.75
LANE_DEV_K = 0.20

NPC_PROXIMITY_K = 0.20
NPC_SAFE_DISTANCE_M = 8.0

DEBUG_REWARD = os.environ.get("DEBUG_REWARD", "0") == "1"


def compute_reward(
    forward_speed,
    lateral,
    npc_dist,
    collided,
    off_road,
    steer=0.0,
    prev_steer=0.0,
):

    progress_reward = PROGRESS_K * max(min(forward_speed, 15.0), 0.0)

    if forward_speed < STATIC_SPEED_THRESHOLD:
        speed_reward = -STATIC_PENALTY
    else:
        speed_dev = abs(forward_speed - TARGET_SPEED_MS) / TARGET_SPEED_MS
        speed_reward = SPEED_K * (1.0 - min(speed_dev, 1.0))

    speed_scale = min(max(forward_speed, 0.0) / TARGET_SPEED_MS, 1.0)
    steer_penalty = -STEER_K * abs(steer) * speed_scale
    delta_steer_penalty = -STEER_RATE_K * (steer - prev_steer) ** 2 * speed_scale

    lat_norm = min(abs(lateral) / LANE_HALF_WIDTH_M, 2.0)
    lane_penalty = -LANE_DEV_K * lat_norm * lat_norm * speed_scale

    if npc_dist < NPC_SAFE_DISTANCE_M:
        npc_penalty = -NPC_PROXIMITY_K * (1.0 - npc_dist / NPC_SAFE_DISTANCE_M)
    else:
        npc_penalty = 0.0

    off_road_penalty = -TERMINAL_PENALTY if (collided or off_road) else 0.0

    total_reward = (
        progress_reward
        + speed_reward
        + steer_penalty
        + delta_steer_penalty
        + lane_penalty
        + npc_penalty
        + off_road_penalty
    )

    if DEBUG_REWARD:
        print(f"progress:    {progress_reward:.3f}")
        print(f"speed:       {speed_reward:.3f}")
        print(f"steer:       {steer_penalty:.3f}")
        print(f"delta_steer: {delta_steer_penalty:.3f}")
        print(f"lane:        {lane_penalty:.3f}")
        print(f"npc:         {npc_penalty:.3f}")
        print(f"off_road:    {off_road_penalty:.3f}")
        print(f"TOTAL:       {total_reward:.3f}")

    return total_reward
