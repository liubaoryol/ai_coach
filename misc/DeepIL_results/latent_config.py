from dataclasses import dataclass
import torch


@dataclass
class LatentConfig:
  latent_size: int
  discrete_latent: bool

  def get_init_latent(self, states: torch.Tensor) -> torch.Tensor:
    return torch.randint(low=0,
                         high=self.latent_size,
                         size=(len(states), ),
                         dtype=torch.float).reshape(len(states), -1)


class LatentConfigEnv2(LatentConfig):
  def get_init_latent(self, states: torch.Tensor) -> torch.Tensor:
    return torch.rand(size=(len(states), ),
                      dtype=torch.float).reshape(len(states), -1)


env1_latent = LatentConfig(4, True)
env2_latent = LatentConfigEnv2(1, False)

LATENT_CONFIG = {
    "env_name1": env1_latent,
    "env_name2": env2_latent,
}
