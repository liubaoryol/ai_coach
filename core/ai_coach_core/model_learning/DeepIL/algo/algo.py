from .ppo import PPO
from .digail import DIGAIL
from .vae import VAE
from .base import NNExpert, LatentNNExpert

# all the algorithms
ALGOS = {
    'ppo': PPO,
    'digail': DIGAIL,
    'vae': VAE,
}

# all the well-trained algorithms
EXP_ALGOS = {
    'ppo': NNExpert,
    'digail': LatentNNExpert,
    'vae': LatentNNExpert,
}
