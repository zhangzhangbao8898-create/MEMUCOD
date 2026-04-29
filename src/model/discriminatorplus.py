import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    MLP discriminator with the same input/output interface as the original version.
    The hidden dimensions are 512 and 256.
    """

    def __init__(self, msk_shape: int, use_sigmoid: bool = True, p_dropout: float = 0.2):
        super().__init__()
        in_dim = 4 * (msk_shape ** 2)

        layers = [
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p_dropout),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p_dropout),

            nn.Linear(256, 1),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.size(0), -1)
        validity = self.model(x_flat)
        return validity
