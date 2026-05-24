import random
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import carla
import cv2
import numpy as np
import yaml

from .observation import ObservationPipeline
from .frame_stack import FrameStack
from .reward import compute_reward
from .scenario_manager import ScenarioManager
from .weather_manager import apply_weather


_RECORD_CAM_TRANSFORMS = {
    "chase": carla.Transform(
        carla.Location(x=-6.0, z=3.0),
        carla.Rotation(pitch=-15.0),
    ),
    "front": carla.Transform(carla.Location(x=2.2, z=1.0)),
    "top": carla.Transform(
        carla.Location(z=25.0),
        carla.Rotation(pitch=-90.0),
    ),
}


class CarlaEnv(gym.Env):

    def __init__(
        self,
        global_step=0,
        record_video=False,
        video_dir="logs/videos",
        video_angles=("chase",),
        video_resolution=(480, 270),
        video_fps=20,
    ):

        super().__init__()

        with open("configs/environment.yaml") as f:
            cfg = yaml.safe_load(f)

        self.cam_w = cfg["camera"]["width"]
        self.cam_h = cfg["camera"]["height"]
        self.cam_fov = cfg["camera"]["fov"]
        self.fixed_dt = cfg.get("fixed_delta_seconds", 0.05)
        self.max_episode_steps = cfg.get("max_episode_steps", 1000)

        self.client = carla.Client(cfg["host"], cfg["port"])
        self.client.set_timeout(20.0)

        self.obs_pipe = ObservationPipeline()
        self.frame_stack = FrameStack()
        self.scenario = ScenarioManager("configs/curriculum.yaml")

        self.global_step = global_step
        self.episode_step = 0
        self.collided = False
        self.current_town = None

        self.world = None
        self.actors = []
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.image = None

        self.record_video = bool(record_video)
        self.video_dir = Path(video_dir)
        self.video_angles = tuple(
            a for a in video_angles if a in _RECORD_CAM_TRANSFORMS
        )
        self.video_w, self.video_h = video_resolution
        self.video_fps = int(video_fps)
        self.rec_cameras = []
        self.rec_frames = {}
        self.video_writer = None
        self.video_episode = 0
        self.video_run_tag = time.strftime("%Y%m%d_%H%M%S")

        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 1, (4, 84, 84), dtype=np.float32),
            "state": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        })

        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
        )

    def _destroy_actors(self):
        if not self.actors:
            self.image = None
            return

        for a in self.actors:
            try:
                a.stop()
            except Exception:
                pass

        try:
            self.client.apply_batch(
                [carla.command.DestroyActor(a.id) for a in self.actors]
            )
        except Exception:
            pass

        self.actors = []
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.image = None
        self.rec_cameras = []
        self.rec_frames = {}

    def _make_record_listener(self, name):
        def cb(data):
            arr = np.frombuffer(data.raw_data, dtype=np.uint8)
            frame = arr.reshape((data.height, data.width, 4))[:, :, :3]
            self.rec_frames[name] = frame.copy()
        return cb

    def _spawn_recording_cameras(self, bp_lib):
        if not self.record_video or not self.video_angles:
            return
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.video_w))
        cam_bp.set_attribute("image_size_y", str(self.video_h))
        cam_bp.set_attribute("fov", "100")
        for name in self.video_angles:
            cam = self.world.spawn_actor(
                cam_bp,
                _RECORD_CAM_TRANSFORMS[name],
                attach_to=self.vehicle,
            )
            self.actors.append(cam)
            self.rec_cameras.append(cam)
            cam.listen(self._make_record_listener(name))

    def _composite_dims(self):
        n = len(self.video_angles)
        if n <= 1:
            return self.video_w, self.video_h
        if n <= 3:
            return self.video_w * n, self.video_h
        return self.video_w * 2, self.video_h * 2

    def _open_video(self):
        if not self.record_video or not self.video_angles:
            return
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.video_episode += 1
        out_w, out_h = self._composite_dims()
        path = self.video_dir / (
            f"{self.video_run_tag}_ep{self.video_episode:03d}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(path), fourcc, self.video_fps, (out_w, out_h)
        )

    def _close_video(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def _write_video_frame(self):
        if self.video_writer is None:
            return
        tiles = []
        for name in self.video_angles:
            frame = self.rec_frames.get(name)
            if frame is None:
                tiles.append(
                    np.zeros((self.video_h, self.video_w, 3), dtype=np.uint8)
                )
                continue
            if frame.shape[0] != self.video_h or frame.shape[1] != self.video_w:
                frame = cv2.resize(frame, (self.video_w, self.video_h))
            tiles.append(frame)
        n = len(tiles)
        if n == 1:
            composite = tiles[0]
        elif n <= 3:
            composite = np.hstack(tiles)
        else:
            row1 = np.hstack(tiles[:2])
            row2 = np.hstack(tiles[2:4])
            composite = np.vstack([row1, row2])
        self.video_writer.write(composite)

    def _setup_sync(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_dt
        self.world.apply_settings(settings)

        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)

    def _spawn_traffic(self, count, npc_spawns):
        if count <= 0:
            return

        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        tm = self.client.get_trafficmanager()

        spawned = 0
        for sp in npc_spawns:
            if spawned >= count:
                break
            bp = random.choice(vehicle_bps)
            npc = self.world.try_spawn_actor(bp, sp)
            if npc is not None:
                npc.set_autopilot(True, tm.get_port())
                self.actors.append(npc)
                spawned += 1

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self._close_video()
        self._destroy_actors()

        town, weather, traffic_count = self.scenario.select(self.global_step)

        if town != self.current_town or self.world is None:
            self.world = self.client.load_world(town)
            self.current_town = town

        self._setup_sync()
        apply_weather(self.world, weather)

        bp_lib = self.world.get_blueprint_library()

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        ego_spawn = spawn_points[0]

        bp = bp_lib.filter("model3")[0]
        self.vehicle = self.world.spawn_actor(bp, ego_spawn)
        self.actors.append(self.vehicle)

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cam_w))
        cam_bp.set_attribute("image_size_y", str(self.cam_h))
        cam_bp.set_attribute("fov", str(self.cam_fov))

        self.image = None
        self.camera = self.world.spawn_actor(
            cam_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=self.vehicle,
        )
        self.actors.append(self.camera)
        self.camera.listen(lambda data: setattr(self, "image", data))

        col_bp = bp_lib.find("sensor.other.collision")
        self.collided = False
        self.collision_sensor = self.world.spawn_actor(
            col_bp,
            carla.Transform(),
            attach_to=self.vehicle,
        )
        self.actors.append(self.collision_sensor)
        self.collision_sensor.listen(lambda e: setattr(self, "collided", True))

        self._spawn_recording_cameras(bp_lib)

        npc_spawns = list(spawn_points[1:])
        random.shuffle(npc_spawns)
        self._spawn_traffic(traffic_count, npc_spawns)

        for _ in range(40):
            self.world.tick()
            if self.image is not None:
                break
        if self.image is None:
            raise RuntimeError("Camera produced no image after 40 ticks")

        self.episode_step = 0

        self._open_video()
        self._write_video_frame()

        obs = self._get_obs()
        obs["image"] = self.frame_stack.reset(obs["image"])

        return obs, {}

    def _get_obs(self):
        img = self.obs_pipe.process_image(self.image)
        state = self.obs_pipe.vector_state(self.vehicle)
        return {"image": img, "state": state}

    def _is_off_road(self):
        loc = self.vehicle.get_location()
        wp = self.world.get_map().get_waypoint(
            loc, project_to_road=False, lane_type=carla.LaneType.Driving
        )
        return wp is None

    def step(self, action):

        steer, throttle, brake = action

        control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
        )
        self.vehicle.apply_control(control)
        self.world.tick()

        self._write_video_frame()

        obs = self._get_obs()
        obs["image"] = self.frame_stack.append(obs["image"])

        off_road = self._is_off_road()
        terminated = bool(self.collided) or off_road
        reward = compute_reward(self.vehicle, self.world, self.collided, off_road)

        self.episode_step += 1
        self.global_step += 1
        truncated = self.episode_step >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def close(self):
        self._close_video()
        self._destroy_actors()
        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                tm = self.client.get_trafficmanager()
                tm.set_synchronous_mode(False)
            except Exception:
                pass
