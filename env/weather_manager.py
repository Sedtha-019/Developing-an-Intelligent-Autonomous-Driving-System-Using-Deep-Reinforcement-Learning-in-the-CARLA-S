import carla

def apply_weather(world, weather_name):
    world.set_weather(getattr(carla.WeatherParameters, weather_name))