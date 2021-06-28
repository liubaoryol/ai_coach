import random
import numpy as np
# from maxent_irl.mdp import (
#     soft_value_iteration, soft_q_value, 
#     )

SMALL_NUMBER = float("-inf")
def soft_q_value(mdp, reward_fn, state, action, v_soft, gamma):
    res = reward_fn(state, action)
    for p, s_prime in mdp.T(state, action):
        next_value = 0
        if not mdp.is_terminal(s_prime) and not v_soft[int(s_prime)] == SMALL_NUMBER:
            next_value = v_soft[int(s_prime)]

        res += p * gamma * next_value
    return res

def soft_value_iteration(mdp, reward_fn, gamma, epsilon=0.001):
    Vsoft = np.full((mdp.num_states,), SMALL_NUMBER)
    count = 0
    while True:
        V_prime = np.zeros((mdp.num_states,))
        for state in range(mdp.num_states):
            if not mdp.is_terminal(state):
                V_prime[state] = SMALL_NUMBER

        # U = U1.copy()
        for state in range(mdp.num_states):
            # if state % 1000 == 0:
            #     print("check running-" + str(state))
            if not mdp.is_terminal(state):
                for action in mdp.legal_actions(state):
                    # print("check running-" + str(action))
                    qval = soft_q_value(mdp, reward_fn, state, action, Vsoft, gamma)
                    V_prime[state] = np.log(np.exp(V_prime[state]) + np.exp(qval))

        count += 1

        delta = np.max(np.abs(Vsoft -  V_prime))
        Vsoft = V_prime
        if delta <= epsilon:
            # print(count)
            return Vsoft

class CMaxEntIRL():
    def __init__(
            self, trajectories, mdp, feature_extractor, gamma=0.9,
            initial_prop=None, learning_rate=0.01, decay=0.001, epsilon=0.001, horizon=100):
        self.feature_extractor = feature_extractor
        self.mdp = mdp
        self.weights = None
        self.gamma = gamma
        self.alpha = learning_rate
        self.decay = decay
        self.eps = epsilon
        self.horizon = horizon
        self.iteration = 0
        self.trajectories = trajectories
        # self.n_iter = num_iter
        # value_fn = value_iteration(mdp, 0.001)
        # self.pi_true_deter = best_policy(mdp, value_fn)
        self.pi_est = None

        self.initial_prop = np.zeros((mdp.num_states))
        if initial_prop is not None:
            self.initial_prop = initial_prop
        else:
            n_states = self.mdp.num_states
            for state in range(n_states):
                self.initial_prop[state] = 1.0 / n_states
        # self.is_terminal = is_terminal
        # if self.is_terminal is None:
        #     self.is_terminal = lambda state: True

    def init_weights(self):
        feature = self.feature_extractor(self.mdp, 0, 0) # feature will be F by 1 array
        self.weights = np.zeros(feature.shape)
        for idx in range(len(feature)):
            self.weights[idx] = random.uniform(0, 1)

    def get_weights(self):
        return self.weights

    def update_weights(self):
        gradient = self.get_gradient(self.trajectories)
        self.weights = (
            self.weights +
            self.alpha / (1 + self.decay * self.iteration) * gradient)

    def calc_empirical_feature_cnt(self, trajectory_set):
        num_traj = len(trajectory_set)
        feature_bar = np.zeros(self.weights.shape)
        for traj in trajectory_set:
            feat_cnt = self.get_avg_feature_counts(traj)
            feature_bar = feature_bar + feat_cnt

        feature_bar = feature_bar / num_traj

        return feature_bar

    def get_gradient(self, trajectory_set):
        empirical_feature_cnt = self.calc_empirical_feature_cnt(trajectory_set)
        optimal_feature_cnt = self.calc_optimal_feature_cnt(self.eps)

        gradient = empirical_feature_cnt - optimal_feature_cnt

        return gradient

    def get_avg_feature_counts(self, trajectory):
        feat_cnt = np.zeros(self.weights.shape)
        counts = len(trajectory)
        for state, action in trajectory:
            feature = self.feature_extractor(self.mdp, state, action)
            feat_cnt = feat_cnt + feature
        feat_cnt = feat_cnt / counts

        return feat_cnt

    def reward(self, state, action):
        feat = self.feature_extractor(self.mdp, state, action)
        return np.dot(self.weights, feat)

    def state_frequency(self):

        d_s_sum = np.array(self.initial_prop)
        d_s_t = np.array(self.initial_prop)

        for dummy in range(self.horizon):
            d_s_tn = np.zeros(d_s_t.shape)
            # delta = 0
            for state in range(self.mdp.num_states):
                for action in self.mdp.legal_actions(state):
                    for prop, s_prime in self.mdp.T(state, action):
                        s_prime = int(s_prime)
                        if self.mdp.is_terminal(s_prime):  # TODO: 
                            continue
                        visit = (
                            d_s_t[state] * self.policy(state, action) * prop)
                        d_s_tn[s_prime] += visit
                        d_s_sum[s_prime] += visit
                # delta = max(delta, abs(d_s_t[state] - d_s_tn[state]))
            d_s_t = d_s_tn

        d_s_sum = d_s_sum / np.sum(d_s_sum)

        return d_s_sum

    def compute_stochastic_pi(self, epsilon):
        v_soft = soft_value_iteration(self.mdp, self.reward, self.gamma, epsilon)

        pi = np.zeros((self.mdp.num_states, self.mdp.num_actions))
        for state in range(self.mdp.num_states):
            vval = v_soft[state]
            for action in self.mdp.legal_actions(state):
                qval = soft_q_value(
                    self.mdp, self.reward, state, action, v_soft, self.gamma)
                pi[state, action] = np.exp(qval - vval)
        self.pi_est = pi

    def policy(self, state, action):
        return self.pi_est[state, action]

    def calc_optimal_feature_cnt(self, epsilon_val_iter):
        self.compute_stochastic_pi(epsilon_val_iter)
        # need some check code for pi
        
        d_s = self.state_frequency()
        # need some check code for d_s
        feat_cnt = np.zeros(self.weights.shape)
        for state in range(self.mdp.num_states):
            d_s_cur = d_s[state]
            for action in self.mdp.legal_actions(state):
                feat = self.feature_extractor(self.mdp, state, action)
                feat_cnt = feat_cnt + feat * d_s_cur * self.policy(state, action)
                
        return feat_cnt


    def do_inverseRL(self, epsilon=0.001, n_max_run=100, callback_reward_pi=None):
        self.init_weights()
        delta = np.inf
        # rel_freq = self.compute_relative_freq()
        # reward_error = []
        # policy_error = []
        self.iteration = 0
        while delta > epsilon and self.iteration < n_max_run:
            weights_old = self.weights.copy()
            self.update_weights()
            if callback_reward_pi:
                callback_reward_pi(self.reward, self.policy)
            # reward_error.append(self.get_reward_error())
            # policy_error.append(self.get_policy_error(rel_freq))
            diff = np.max(np.abs(weights_old - self.weights))

            delta = diff
            self.iteration += 1
            print("Delta-" + str(delta) + ", cnt-" + str(self.iteration))
            # print(self.weights)


def compute_relative_freq(mdp, trajectories):
    rel_freq = np.zeros(mdp.num_states)
    count = 0
    for traj in trajectories:
        for s, a in traj:
            rel_freq[s] += 1
            count += 1
        
    rel_freq = rel_freq / count
    # for state in rel_freq:
    #     rel_freq[state] = rel_freq[state] / count

    return rel_freq

def cal_reward_error(mdp, fn_reward_irl):
    sum_diff = 0
    for s_idx in range(mdp.num_states):
        for a_idx in mdp.legal_actions(s_idx):
            r_irl = fn_reward_irl(s_idx, a_idx)
            r_mdp = mdp.reward(s_idx, a_idx, None)
            sum_diff += (r_irl - r_mdp) ** 2

    return np.sqrt(sum_diff)

def cal_policy_error(rel_freq, mdp, pi_irl, pi_true):
    def compute_kl(s_idx):
        sum_val = 0
        for a_idx in mdp.legal_actions(s_idx):
            p_irl = pi_irl(s_idx, a_idx)
            p_h = pi_true[s_idx, a_idx]
            if p_irl != 0 and p_h != 0:
                sum_val += p_irl * (np.log(p_irl) - np.log(p_h))
        return sum_val

    sum_kl = 0
    for s_idx in range(mdp.num_states):
        sum_kl += rel_freq[s_idx] * compute_kl(s_idx)

    return sum_kl

# if __name__ == "__main__":

#     trans = get_transition_p2()

#     mdp = CMDP_P2(trans, 0.9)
#     value_fn = value_iteration(mdp, 0.001)
#     pi = best_policy(mdp, value_fn)
#     sto_pi = get_stochastic_policy(mdp, pi)

#     trajectories = []
#     len_sum = 0
#     file_names = glob.glob(os.path.join(DATA_DIR, '*.txt'))
#     for file_nm in file_names:
#         traj = read_trajectory(file_nm)
#         len_sum += len(traj)
#         trajectories.append(traj)
    
#     avg_len = int(len_sum / len(trajectories)) + 1
#     print(avg_len)

#     init_prop = {s: 0 for s in mdp.states()}
#     init_prop[(0, 0, 0)] = 1

#     rel_freq = compute_relative_freq(mdp, trajectories)
#     reward_error = []
#     policy_error = []
#     def compute_errors(reward_fn, policy_fn):
#         reward_error.append(cal_reward_error(mdp, reward_fn))
#         policy_error.append(cal_policy_error(rel_freq, mdp, policy_fn, sto_pi))
        
#     use_original_feature_set = False
#     if use_original_feature_set:
#         irl = CMaxEntIRL(trajectories, mdp, feature_extractor=feature_extract, initial_prop=init_prop, horizon=avg_len)
#         irl.do_inverseRL(epsilon=0.001, n_max_run=1000, callback_reward_pi=compute_errors)

#         with open('reward_error.pickle', 'wb') as f:
#             pickle.dump(reward_error, f, pickle.HIGHEST_PROTOCOL) 
#         with open('policy_error.pickle', 'wb') as f:
#             pickle.dump(policy_error, f, pickle.HIGHEST_PROTOCOL) 
#     else:
#         irl = CMaxEntIRL(trajectories, mdp, feature_extractor=feature_extract_updated, initial_prop=init_prop, horizon=avg_len)
#         irl.do_inverseRL(epsilon=0.0001, n_max_run=1000, callback_reward_pi=compute_errors)

#         with open('reward_error_updated.pickle', 'wb') as f:
#             pickle.dump(reward_error, f, pickle.HIGHEST_PROTOCOL) 
#         with open('policy_error_updated.pickle', 'wb') as f:
#             pickle.dump(policy_error, f, pickle.HIGHEST_PROTOCOL) 
