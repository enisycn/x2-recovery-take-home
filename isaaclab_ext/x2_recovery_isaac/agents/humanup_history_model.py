"""HumanUP's 10-step history encoder adapted to RSL-RL 5 model APIs."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


class StateHistoryEncoder(nn.Module):
    """The exact 10-step projection/Conv1d shape in HumanUP's release."""

    def __init__(self, input_size: int, output_size: int, activation: str) -> None:
        super().__init__()
        activation_layer = resolve_nn_activation(activation)
        self.encoder = nn.Sequential(nn.Linear(input_size, 30), activation_layer)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(30, 20, kernel_size=4, stride=2),
            resolve_nn_activation(activation),
            nn.Conv1d(20, 10, kernel_size=2, stride=1),
            resolve_nn_activation(activation),
            nn.Flatten(),
        )
        self.linear_output = nn.Sequential(nn.Linear(30, output_size), resolve_nn_activation(activation))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch, steps, features = history.shape
        projected = self.encoder(history.reshape(batch * steps, features))
        encoded = self.conv_layers(projected.reshape(batch, steps, 30).permute(0, 2, 1))
        return self.linear_output(encoded)


class HumanUpHistoryActor(MLPModel):
    """Actor using current proprioception plus a learned history latent."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        proprio_dim: int = 98,
        history_length: int = 10,
        history_latent_dim: int = 20,
    ) -> None:
        self.proprio_dim = proprio_dim
        self.history_length = history_length
        self.history_latent_dim = history_latent_dim
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        expected = proprio_dim + (4 + 1 + 2 * output_dim + 3) + history_length * proprio_dim
        if self.obs_dim != expected:
            raise ValueError(f"HumanUP observation width must be {expected}, got {self.obs_dim}")
        self.history_encoder = StateHistoryEncoder(proprio_dim, history_latent_dim, activation)

    def _get_latent_dim(self) -> int:
        return self.proprio_dim + self.history_latent_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        full = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        full = self.obs_normalizer(full)
        proprio = full[:, : self.proprio_dim]
        history = full[:, -self.history_length * self.proprio_dim :]
        history = history.reshape(-1, self.history_length, self.proprio_dim)
        return torch.cat((proprio, self.history_encoder(history)), dim=1)


class HumanUpInferenceModule(nn.Module):
    """Tensor-only deployment view of :class:`HumanUpHistoryActor`.

    RSL-RL's generic exporter bypasses ``get_latent`` and feeds the complete
    1,148-value observation directly to the 118-input MLP.  This wrapper keeps
    the trained normalizer and history encoder in the exported graph, so the
    deployment contract exactly matches deterministic training inference.
    """

    def __init__(self, actor: HumanUpHistoryActor) -> None:
        super().__init__()
        self.obs_normalizer = actor.obs_normalizer
        self.history_encoder = actor.history_encoder
        self.mlp = actor.mlp
        self.proprio_dim = actor.proprio_dim
        self.history_length = actor.history_length

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = self.obs_normalizer(observation)
        proprio = normalized[:, : self.proprio_dim]
        history = normalized[:, -self.history_length * self.proprio_dim :]
        history = history.reshape(-1, self.history_length, self.proprio_dim)
        latent = torch.cat((proprio, self.history_encoder(history)), dim=1)
        return self.mlp(latent)
