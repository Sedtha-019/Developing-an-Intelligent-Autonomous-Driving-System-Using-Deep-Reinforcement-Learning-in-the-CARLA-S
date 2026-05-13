import random

import gymnasium as gym
from gymnasium import spaces
import carla
import numpy as np
import yaml

from .observation import ObservationPipeline
from .frame_stack import FrameStack
from .reward import compute_reward
from .scenario_manager import ScenarioManager
from .weather_manager import apply_weather


class CarlaEnv(gym.Env):

    def __init__(self, global_step=0):

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
