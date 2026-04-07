# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalPointNetEncoder(nn.Module):
    def __init__(
        self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None
    ):
        super().__init__()
        layers = []
        for i in range(num_pre_layers):
            layers.append(nn.Linear(in_channels if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        self.pre_mlps = nn.Sequential(*layers)
        layers = []
        for i in range(num_layers - num_pre_layers):
            layers.append(
                nn.Linear(hidden_dim * 2 if i == 0 else hidden_dim, hidden_dim)
            )
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        self.mlps = nn.Sequential(*layers)

        if out_channels is not None:
            layers = []
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_channels))
            self.out_mlps = nn.Sequential(*layers)
        else:
            self.out_mlps = None

    def forward(self, polylines):
        """
        Args:
            polylines: [B, A, T, C] batch of polylines per agent.

        Returns:
            Tensor of shape [B, A, D] with D == hidden_dim (pooled over T).
        """
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape
        polylines = rearrange(polylines, "b a t d -> (b a) t d")
        # pre-mlp
        polylines = rearrange(polylines, "ba t d -> (ba t) d")
        polylines_feature = self.pre_mlps(polylines)
        polylines_feature = rearrange(
            polylines_feature, "(ba t) d -> ba d t", t=num_points_each_polylines
        )
        # get Lookback global feature
        polylines_feature_padded = F.pad(
            polylines_feature,
            (num_points_each_polylines - 1, 0),
            "constant",
            0,
        )
        pooled_feature = F.max_pool1d(
            polylines_feature_padded,
            kernel_size=num_points_each_polylines,
            stride=1,
        )  # [ba, d, t]
        polylines_feature = torch.cat((polylines_feature, pooled_feature), dim=-2)

        # mlp
        polylines_feature = rearrange(polylines_feature, "ba d t -> (ba t) d")
        feature_buffers = self.mlps(polylines_feature)
        feature_buffers = rearrange(
            feature_buffers, "(ba t) d -> ba d t", t=num_points_each_polylines
        )

        # max-pooling
        feature_buffers = F.max_pool1d(
            feature_buffers,
            kernel_size=num_points_each_polylines,
            stride=1,
        )

        feature_buffers = feature_buffers.squeeze(-1)
        feature_buffers = rearrange(
            feature_buffers,
            "(b a) d -> b a d",
            b=batch_size,
            a=num_polylines,
        )

        return feature_buffers


if __name__ == "__main__":
    encoder = CausalPointNetEncoder(
        in_channels=2, hidden_dim=4, num_layers=3, num_pre_layers=1, out_channels=4
    )
    encoder.eval()
    polylines = torch.rand(16, 11, 20, 2)
    feature_buffers = encoder(polylines)
    print(feature_buffers.shape)
