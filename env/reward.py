def compute_reward(vehicle, world, collided, off_road):

    vel = vehicle.get_velocity()
    transform = vehicle.get_transform()
    fwd = transform.get_forward_vector()

    forward_speed = vel.x * fwd.x + vel.y * fwd.y
    forward_speed = max(min(forward_speed, 15.0), -2.0)

    loc = vehicle.get_location()
    wp = world.get_map().get_waypoint(loc, project_to_road=True)
    lat_dist = loc.distance(wp.transform.location) if wp else 2.0

    reward = 0.05 * forward_speed - 0.1 * lat_dist

    if collided or off_road:
        reward -= 50.0

    return reward
