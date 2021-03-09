import numpy as np


def conv_to_np_env(np_bags, agent1, agent2):
    np_env = np.copy(np_bags)
    a1_pos = agent1.coord
    a2_pos = agent2.coord
    a1_hold = agent1.hold
    a2_hold = agent2.hold 

    # empty: 0 / bag: 1 / a1: 2 / a2: 3 / a1&a2: 4
    # bag&a1: 5 / bag&a1h: 6 / bag&a2: 7 / bag&a2h: 8
    # bag&a1&a2: 9 / bag&a1h&a2: 10 / bag&a1&a2h: 11 / bag&a1h&a2h: 12
    if a1_pos == a2_pos:
        is_bag = np_env[a1_pos] == 1
        if is_bag:
            if a1_hold and a2_hold:
                np_env[a1_pos] = 12
            elif a1_hold:
                np_env[a1_pos] = 10
            elif a2_hold:
                np_env[a1_pos] = 11
            else:
                np_env[a1_pos] = 9
        else:
            np_env[a1_pos] = 4
    else:
        if np_env[a1_pos] == 1:
            np_env[a1_pos] = 6 if a1_hold else 5
        else:
            np_env[a1_pos] = 2
        
        if np_env[a2_pos] == 1:
            np_env[a2_pos] = 8 if a2_hold else 7
        else:
            np_env[a2_pos] = 3
    return np_env.ravel()