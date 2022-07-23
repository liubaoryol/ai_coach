from .ppo import PPO, PPOExpert
from .digail import DIGAIL, DIGAILExpert

# all the algorithms
ALGOS = {
    'ppo': PPO,
    'digail': DIGAIL,
}

# all the well-trained algorithms
EXP_ALGOS = {
    'ppo': PPOExpert,
    'digail': DIGAILExpert,
}
