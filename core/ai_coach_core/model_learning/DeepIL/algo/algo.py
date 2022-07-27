from .ppo import PPO
from .digail import DIGAIL
from .base import NNExpert

# all the algorithms
ALGOS = {
    'ppo': PPO,
    'digail': DIGAIL,
}

# all the well-trained algorithms
EXP_ALGOS = {
    'ppo': NNExpert,
    'digail': NNExpert,
}
