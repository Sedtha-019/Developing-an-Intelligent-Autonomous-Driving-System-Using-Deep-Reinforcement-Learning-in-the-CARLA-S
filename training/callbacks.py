from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):

    def __init__(self, save_freq, save_path):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save = 0

    def _on_training_start(self):
        self.last_save = self.num_timesteps

    def _on_step(self):
        if self.num_timesteps - self.last_save >= self.save_freq:
            self.last_save = self.num_timesteps
            self.model.save(f"{self.save_path}/model_{self.num_timesteps}")
            self.training_env.save(f"{self.save_path}/vecnormalize.pkl")
        return True
