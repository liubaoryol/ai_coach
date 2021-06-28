from enum import Enum


NUM_X_GRID = 8
NUM_Y_GRID = 8
KEY_BAGS = 'bags'
KEY_AGENTS = 'agents'
KEY_INPUT = 'input'
KEY_TIMER = 'timer'
KEY_USERNAME = 'username'
KEY_STEPS = 'steps'
KEY_A1_POS = 'a1_pos'
KEY_A2_POS = 'a2_pos'
KEY_A1_HOLD = 'a1_hold'
KEY_A2_HOLD = 'a2_hold'

LATENT_HEAVY_BAGS = 0
LATENT_LIGHT_BAGS = 1
UNKNOWN_LATENT = None

class AgentActions(Enum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    HOLD = 5  # toggle
