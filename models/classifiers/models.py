import torch
import torch.nn as nn

class KeypointMLP(nn.Module):
    def __init__(self, num_keypoints: int, hidden_dim: int = 256, out_dim: int = 10):
        super().__init__()
        in_dim = num_keypoints * 3  # (x, y, mask) for each keypoint

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, keypoints: torch.Tensor, mask: torch.Tensor):
        """
        keypoints: (B, K, 2)
        mask:      (B, K) with 1.0 = valid, 0.0 = ignore
        """
        # Zero-out invalid keypoints (optional but common)
        keypoints = keypoints * mask.unsqueeze(-1)  # (B, K, 2)

        # Concatenate mask as a feature channel
        mask_feat = mask.unsqueeze(-1)              # (B, K, 1)
        x = torch.cat([keypoints, mask_feat], dim=-1)  # (B, K, 3)

        # Flatten the keypoint dimension into the feature dimension
        x = x.view(x.size(0), -1)  # (B, K*3)

        # Run through MLP
        out = self.net(x)          # (B, out_dim)
        return out

class KeypointMLPDeeper(nn.Module):
    def __init__(self, num_keypoints: int,
                 hidden_dim: int = 256,
                 out_dim: int = 10,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        in_dim = num_keypoints * 3  # (x, y, mask)

        layers = []
        dim_in = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim_in = hidden_dim

        layers.append(nn.Linear(hidden_dim, out_dim))  # logits

        self.net = nn.Sequential(*layers)

    def forward(self, keypoints: torch.Tensor, mask: torch.Tensor):
        keypoints = keypoints * mask.unsqueeze(-1)
        mask_feat = mask.unsqueeze(-1)
        x = torch.cat([keypoints, mask_feat], dim=-1)
        x = x.view(x.size(0), -1)
        return self.net(x)