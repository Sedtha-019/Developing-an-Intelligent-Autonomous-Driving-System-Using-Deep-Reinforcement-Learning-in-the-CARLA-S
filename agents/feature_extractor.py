import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiModalExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space):

        super().__init__(observation_space, features_dim=256)

        n_input_channels = observation_space["image"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space["image"].shape)
            cnn_out = self.cnn(sample).shape[1]

        state_dim = observation_space["state"].shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 64, 256),
            nn.ReLU(),
        )

    def forward(self, obs):
        img = obs["image"]
        vec = obs["state"]
        cnn_feat = self.cnn(img)
        vec_feat = self.mlp(vec)
        return self.fusion(torch.cat([cnn_feat, vec_feat], dim=1))
