import numpy as np
from policy.mdp import MDP
from policy.mdp_utils import (
    StateSpace, ActionSpace)
from moving_luggage.constants import (
    NUM_X_GRID, NUM_Y_GRID, AgentActions)


class MDPMovingLuggage(MDP):
    def __init__(self):
        self.num_x_grid = NUM_X_GRID
        self.num_y_grid = NUM_Y_GRID
        super().__init__()
 
    def init_statespace(self):
        """Defines MDP state space.
        """
        self.dict_factored_statespace = {}
        s_dist_goal = StateSpace(range(self.num_x_grid * self.num_y_grid))
        s_dist_target = StateSpace(range(self.num_x_grid + self.num_y_grid))
        s_dist_agent = StateSpace(range(self.num_x_grid + self.num_y_grid))
        s_on_bag = StateSpace([0, 1])
        s_hold = StateSpace([0, 1, 2])
        # s_both_hold = StateSpace([0, 1])
        self.dict_factored_statespace = {
            0: s_dist_goal, 1: s_dist_target, 2: s_dist_agent,
            3: s_on_bag, 4: s_hold}
        
    def init_actionspace(self):
        """Defines MDP action space.
        """
        self.dict_factored_actionspace = {}
        a_space = ActionSpace(AgentActions)
        self.dict_factored_actionspace = {0: a_space}

    def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
        pass

    def reward(self, state_idx: int, action_idx: int) -> float:
        pass


class MDPMovingLuggage_V2(MDP):
    def __init__(self):
        self.num_x_grid = NUM_X_GRID
        self.num_y_grid = NUM_Y_GRID
        super().__init__()
 
    def init_statespace(self):
        """Defines MDP state space.
        """
        self.dict_factored_statespace = {}
        s_dir_agent = StateSpace(range(9))
        s_dist_agent = StateSpace([0, 1])
        s_dir_target = StateSpace(range(16))
        s_dir_goal = StateSpace(range(5))
        s_on_bag = StateSpace([0, 1])
        s_hold = StateSpace([0, 1, 2])
        # s_both_hold = StateSpace([0, 1])
        self.dict_factored_statespace = {
            0: s_dir_agent, 1: s_dist_agent, 2: s_dir_target,
            3: s_dir_goal, 4: s_on_bag, 5: s_hold}
        
    def init_actionspace(self):
        """Defines MDP action space.
        """
        self.dict_factored_actionspace = {}
        a_space = ActionSpace(AgentActions)
        self.dict_factored_actionspace = {0: a_space}

    def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
        pass

    def reward(self, state_idx: int, action_idx: int) -> float:
        pass