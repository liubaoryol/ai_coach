# import argparse
import random
import os, glob
import numpy as np

TERMINAL_STATE = "terminal"
# Retrieved from COMP540 assignment 
class CBaseMDP:
    def __init__(self, transitions, gamma):
        self.transitions = transitions
        self.gamma = gamma

    def reward(self, state, action, next_state):
        pass

    def actions(self, state):
        return list(self.transitions[state].keys())

    def states(self):
        return list(self.transitions.keys())

    def transition(self, state, action):
        tr = self.transitions[state].get(action)
        if tr:
            return tr
        else:
            raise KeyError(str(state) + ": " + str(action))
    
    def is_terminal(self, state):
        pass


class CMDP_P2(CBaseMDP):
    def __init__(self, transitions, gamma):
        super().__init__(transitions, gamma)

    def reward(self, state, action, next_state):
        x, y, h = state
        if (x, y) in DANGER_GRIDS: # terminal state
            return -100

        if h == 0:
            if (x, y) == (3, 3):
                if action == PICK:
                    return 100
                else:
                    return -1
            elif x == 3 or x == 4:
                return -10
            else:
                return -1
        else:
            if (x, y) == (0, 0):  # terminal state
                return 100
            elif x == 0 or x == 1:
                return -10
            else:
                return -1


    def is_terminal(self, state):
        return state == TERMINAL_STATE
        # x, y, h = state
        # if state == (0, 0, 1) or (x, y) in [(1, 3), (4, 1), (4, 2)]:
        #     return True
        # else:
        #     return False

################################################################################
# Codes in this section are retrieved from COMP540 assignment 
def q_value(mdp, s, a, U):
    res = 0
    for p, s_prime in mdp.transition(s, a):
        next_value = 0
        if not mdp.is_terminal(s_prime):
            next_value = U[s_prime]

        res += p * (mdp.reward(s, a, s_prime) + mdp.gamma * next_value)
    return res


# Value Iteration
def value_iteration(mdp, epsilon=0.001):
    """Solving an MDP by value iteration. [Figure 16.6]"""

    U1 = {s: 0 for s in mdp.states()}
    count = 0
    while True:
        count += 1
        U = U1.copy()
        delta = 0
        for s in mdp.states():
            if mdp.is_terminal(s):
                U1[s] = U[s]
            else:
                U1[s] = max(q_value(mdp, s, a, U) for a in mdp.actions(s))
            delta = max(delta, abs(U1[s] - U[s]))
        # if delta <= epsilon * (1 - mdp.gamma) / mdp.gamma:
        if delta <= epsilon:
            # print(count)
            return U

# Policy Iteration
def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action."""

    pi = {}
    for s in mdp.states():
        if mdp.is_terminal(s):
            pi[s] = (0, 0)
        else:
            pi[s] = max(mdp.actions(s), key=lambda a: q_value(mdp, s, a, U))
    return pi


################################################################################
SMALL_NUMBER = float("-inf")
def soft_q_value(mdp, reward_fn, state, action, v_soft):
    res = reward_fn(state, action)
    for p, s_prime in mdp.transition(state, action):
        next_value = 0
        if not mdp.is_terminal(s_prime) and not v_soft[s_prime] == SMALL_NUMBER:
            next_value = v_soft[s_prime]

        res += p * mdp.gamma * next_value
    return res

def soft_value_iteration(mdp, reward_fn, epsilon=0.001):
    Vsoft = {s: SMALL_NUMBER for s in mdp.states()}
    while True:
        V_prime = {s: 0 if mdp.is_terminal(s) else SMALL_NUMBER for s in mdp.states()}
        # U = U1.copy()
        delta = 0
        for state in mdp.states():
            if not mdp.is_terminal(state):
                for action in mdp.actions(state):
                    qval = soft_q_value(mdp, reward_fn, state, action, Vsoft)
                    # res = reward_fn(state, action)
                    # for p, s_prime in mdp.transition(state, action):
                    #     next_value = 0
                    #     if not mdp.is_terminal(s_prime) and not Vsoft[s_prime] == SMALL_NUMBER:
                    #         next_value = Vsoft[s_prime]

                    #     res += p * mdp.gamma * next_value

                    # print(V_prime[state])
                    # print(qval)
                    V_prime[state] = np.log(np.exp(V_prime[state]) + np.exp(qval))

            delta = max(delta, abs(Vsoft[state] - V_prime[state]))
        Vsoft = V_prime.copy()
        if delta <= epsilon:
            # print(count)
            return Vsoft

def print_trans(trans):
    text = ""
    for s in sorted(trans):
        text += repr(s) + "\n"
        for a in trans[s]:
            text += "    " + repr(a) + ": ["
            for t in trans[s][a]:
                text += repr(t) + ", "
            text += "]\n"

    print(text)


def print_pi(pi):
    text = ""
    for s in pi:
        a = pi[s]
        text += repr(s) + ":   " + repr(a) + "\n"
    print(text)


def get_transition_p2():
    set_state = set()
    for i in range(5):
        for j in range(5):
            state = (i, j)
            if state not in [(2, 1), (2, 2), (2, 3)]:
                set_state.add((state[0], state[1], 0))
                set_state.add((state[0], state[1], 1))

    def get_neighborhood(stt):
        set_neighbor = set()
        x, y, h = stt
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                state = (x + i, y + j, h)
                if state in set_state:
                    set_neighbor.add(state)
        return set_neighbor

    actions = set()
    actions.add((-1, 0))
    actions.add((1, 0))
    actions.add((0, -1))
    actions.add((0, 1))
    actions.add((0, 0))
    actions.add((-1, -1))
    actions.add((-1, 1))
    actions.add((1, -1))
    actions.add((1, 1))
    actions.add(PICK)

    trans = {}
    for s_cur in set_state:
        trans[s_cur] = {}
        if s_cur == (0, 0, 1):
            trans[s_cur][(0, 0)] = [(1.0, TERMINAL_STATE)]
            continue
        elif (s_cur[0], s_cur[1]) in DANGER_GRIDS:
            trans[s_cur][(0, 0)] = [(1.0, TERMINAL_STATE)]
            continue
        for act in actions:
            if act == PICK:
                if s_cur == (3, 3, 0):
                    s_m = (3, 3, 1)
                    trans[s_cur][act] = [(1.0, s_m)]
                # else:
                #     s_m = s_cur
            else:
                s_m = (s_cur[0] + act[0], s_cur[1] + act[1], s_cur[2])
                if s_m not in set_state:
                    continue

                l1norm = abs(act[0]) + abs(act[1])
                if l1norm == 0:
                    trans[s_cur][act] = [(1.0, s_m)]
                elif l1norm == 1:
                    trans[s_cur][act] = [(0.9, s_m)]
                    set_neighbor = get_neighborhood(s_cur)
                    len_nbr = len(set_neighbor) - 1
                    for s_nbr in set_neighbor:
                        if s_nbr != s_m:
                            trans[s_cur][act].append((0.1 / len_nbr, s_nbr))
                elif l1norm == 2:
                    trans[s_cur][act] = [(0.7, s_m)]
                    set_neighbor = get_neighborhood(s_cur)
                    len_nbr = len(set_neighbor) - 1
                    for s_nbr in set_neighbor:
                        if s_nbr != s_m:
                            trans[s_cur][act].append((0.3 / len_nbr, s_nbr))
                else:
                    print("Shouldn't fall here") 

    return trans


def get_stochastic_policy(mdp, deter_pi):
    sto_pi = {}
    for state in mdp.states():
        sto_pi[state] = {}
        deter_act = deter_pi[state]
        possible_acts = mdp.actions(state)
        if len(possible_acts) == 1:
            sto_pi[state][deter_act] = 1.0
        else:
            for act in possible_acts:
                if act == deter_act:
                    sto_pi[state][act] = 0.95
                else:
                    sto_pi[state][act] = 0.05 / (len(possible_acts) - 1)

    return sto_pi


def gen_trajectory(mdp, stochastic_pi):

    init_state = (0, 0, 0)

    trajectory = []
    cur_state = init_state
    while True:
        if cur_state == TERMINAL_STATE:
            break

        # select an action
        # list_actions, list_prop = get_stochastic_policy(mdp, deter_policy, cur_state)
        list_actions = []
        list_prop = []
        for action in stochastic_pi[cur_state]:
            list_actions.append(action)
            list_prop.append(stochastic_pi[cur_state][action])

        act = random.choices(list_actions, weights=list_prop, k=1)[0]
        trajectory.append((cur_state, act))

        # transition to next state
        list_next_dist = mdp.transition(cur_state, act)
        list_next_states = []
        list_next_prop = []
        for p, s_next in list_next_dist:
            list_next_states.append(s_next)
            list_next_prop.append(p)

        cur_state = random.choices(list_next_states, weights=list_next_prop, k=1)[0]

    return trajectory

def save_trajectory(trajectory, file_name):
    dir_path = os.path.dirname(file_name)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_name, 'w', newline='') as txtfile:
        # sequence
        txtfile.write('# state action sequence\n')
        for idx in range(len(trajectory)):
            state, action = trajectory[idx]
            txtfile.write('%d, %d, %d, %d, %d' %
            (state[0], state[1], state[2], action[0], action[1]))
            txtfile.write('\n')


def read_trajectory(file_name):
    traj = []
    with open(file_name, newline='') as txtfile:
        lines = txtfile.readlines()
        for i_r in range(1, len(lines)):
            line = lines[i_r]
            row_elem = [int(elem) for elem in line.rstrip().split(", ")]
            state = (row_elem[0], row_elem[1], row_elem[2])
            action = (row_elem[3], row_elem[4])
            traj.append((state, action))
    return traj


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generating results for Problem Set 2")
#     parser.add_argument("--policy", dest="policy", action="store_true", help="show the optimal policy and value")
#     parser.add_argument("--makedata", dest="num_samples", type=int, help="generate trajectory samples")
#     args = parser.parse_args()

#     trans = get_transition_p2()

#     # print_trans(trans)
#     mdp = CMDP_P2(trans, 1)
#     value_fn = value_iteration(mdp, 0.001)
#     # value_fn = soft_value_iteration(mdp, lambda s, a: mdp.reward(s, a, None), 0.001)
#     pi = best_policy(mdp, value_fn)

#     # print(value_fn)
#     # print_pi(pi)

#     # print_trans(trans)
#     # args.policy = True
#     # for state in trans:
#     if args.policy:
#         for state in sorted(mdp.states()):
#             val = value_fn[state]
#             action = pi[state]
#             str_action = "wait"
#             if action == (0, 1):
#                 str_action = "uparrow"
#             elif action == (0, -1):
#                 str_action = "downarrow"
#             elif action == (1, 0):
#                 str_action = "rightarrow"
#             elif action == (-1, 0):
#                 str_action = "leftarrow"
#             elif action == (1, 1):
#                 str_action = "nearrow"
#             elif action == (-1, 1):
#                 str_action = "nwarrow"
#             elif action == (1, -1):
#                 str_action = "searrow"
#             elif action == (-1, -1):
#                 str_action = "swarrow"
#             elif action == PICK:
#                 str_action = "pick"

#             print(str(state) + ": " + str_action + " / " + "%.2f" % (val,))

#     if args.num_samples:
#         if args.num_samples > 0:
#             for dummy in range(args.num_samples):
#                 sto_pi = get_stochastic_policy(mdp, pi)
#                 sample = gen_trajectory(mdp, sto_pi)
#                 file_path = os.path.join(DATA_DIR, str(dummy) + '.txt')
#                 save_trajectory(sample, file_path)
#                 # print(sample)
#             print("data generated")

    

