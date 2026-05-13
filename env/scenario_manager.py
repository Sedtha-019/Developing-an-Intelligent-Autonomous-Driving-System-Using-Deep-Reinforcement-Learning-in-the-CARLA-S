import random
import yaml


class ScenarioManager:

    def __init__(self, path):
        with open(path) as f:
            self.phases = yaml.safe_load(f)["phases"]

    def select(self, step):

        for phase in self.phases:
            if step <= phase["until_step"]:
                return (
                    random.choice(phase["towns"]),
                    random.choice(phase["weather"]),
                    phase["traffic"],
                )

        last = self.phases[-1]
        return (
            random.choice(last["towns"]),
            random.choice(last["weather"]),
            last["traffic"],
        )
