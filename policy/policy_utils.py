import numpy as np
import heapq

from typing_extensions import final
from moving_luggage.constants import AgentActions


ON_BAG = "on_bag"
HOLDING = "holding"
BOTH_HOLD = "both_hold"
DIST_TAR = "dist_target"
DIST_AGENT = "dist_agent"
DIST_GOAL = "dist_goal"

DIR_AGENT = "dir_agent"
DIR_GOAL = "dir_goal"
DIR_TARGET = "dir_target"


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


# retrieved from COMP540 assignment
class PriorityQueue:
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


# retrieved from COMP540 assignment
class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print(a['test'])

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print(a['test'])
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print(a['test'])
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print(a['blah'])
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()
        compare = lambda x, y:  sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y ):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x,y = y,x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__( self, y ):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__( self, y ):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]

        return addend


def manhattan_distance(start, end):
    xy1 = start
    xy2 = end
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


# A* grid world
def get_astar_distance(np_env, start, list_end, hueristic=None):
    def estimate_cost(cur, cost_so_far):
        min_dist = float("inf")
        for end in list_end:
            dist = hueristic(cur, end)
            if min_dist > dist:
                min_dist = dist

        return cost_so_far + min_dist

    num_x, num_y = np_env.shape
    def get_neighbor(pos):
        xxx, yyy = pos
        list_neighbor = []
        x_new = xxx - 1
        if x_new >= 0 and x_new < num_x:
            if np_env[(x_new, yyy)] == 0:
                list_neighbor.append((x_new, yyy))
        x_new = xxx + 1
        if x_new >= 0 and x_new < num_x:
            if np_env[(x_new, yyy)] == 0:
                list_neighbor.append((x_new, yyy))
        y_new = yyy - 1
        if y_new >= 0 and y_new < num_y:
            if np_env[(xxx, y_new)] == 0:
                list_neighbor.append((xxx, y_new))
        y_new = yyy + 1
        if y_new >= 0 and y_new < num_y:
            if np_env[(xxx, y_new)] == 0:
                list_neighbor.append((xxx, y_new))
        
        return list_neighbor

    frontier = PriorityQueue()
    visited = Counter()

    cost_sum = 0
    frontier.push((start, []), estimate_cost(start, cost_sum))
    final_path = []
    found = False
    while not frontier.isEmpty():
        current, path = frontier.pop()
        visited[current] = 1
        if current in list_end:
            final_path = path
            found = True
            break

        cost_sum = len(path)
        for neighbor in get_neighbor(current):
            if visited[neighbor] is not 0:
                continue

            child_item = (neighbor, path + [neighbor])
            cost_est = estimate_cost(neighbor, cost_sum + 1)
            # need refactoring
            for index, (pri, cnt, item) in enumerate(frontier.heap):
                if item[0] == neighbor:
                    if pri <= cost_est:
                        break
                    del frontier.heap[index]
                    frontier.heap.append((cost_est, cnt, child_item))
                    heapq.heapify(frontier.heap)
                    break
            else:
                frontier.push(child_item, cost_est)
    
    if found:
        return len(final_path), final_path
    else:
        return num_x * num_y - 1, final_path



def get_nearest_carrying_distance(np_trgt, agent_pos, goals):
    trgt_pos = np.argwhere(np_trgt)
    min_dist = float("inf")
    for pos in trgt_pos:
        pos_tub = tuple(pos)
        dist_astar, _ = get_astar_distance(
            np_trgt, pos_tub, goals, hueristic=manhattan_distance)
        dist_man = manhattan_distance(agent_pos, pos_tub)
        dist = dist_astar + dist_man
        if min_dist > dist:
            min_dist = dist

    return min_dist

def get_nearest_target(np_trgt, agent_pos):
    trgt_pos = np.argwhere(np_trgt)
    min_dist = float("inf")
    target = None
    for pos in trgt_pos:
        pos_tub = tuple(pos)
        dist_man = manhattan_distance(agent_pos, pos_tub)
        if min_dist > dist_man:
            min_dist = dist_man
            target = pos_tub

    return min_dist, target

def extract_qlearn_features(state, goals):
    np_trgt, a1_pos, a2_pos, a1_hold, a2_hold = state
    dist_a1_trgt = get_nearest_carrying_distance(np_trgt, a1_pos, goals)
    dist_a2_trgt = get_nearest_carrying_distance(np_trgt, a2_pos, goals)
    dist_agents = manhattan_distance(a1_pos, a2_pos)

    dict_features = {}
    dict_features["a1_target"] = dist_a1_trgt
    dict_features["a2_target"] = dist_a2_trgt
    dict_features["dist_agents"] = dist_agents

    dict_features["a1_hold"] = 1 if a1_hold else 0
    dict_features["a2_hold"] = 1 if a2_hold else 0
    dict_features["both_hold"] = 0

    if a1_hold and a2_hold and a1_pos == a2_pos:
        dict_features["a1_hold"] = 0
        dict_features["a2_hold"] = 0
        dict_features["both_hold"] = 1

    return dict_features


def extract_qlearn_features_indv(state, action_idx, agent_idx, goals):
    np_trgt, a1_pos, a2_pos, a1_hold, a2_hold = state

    dict_features = {}
    my_hold = a1_hold
    op_hold = a2_hold
    my_pos= a1_pos
    op_pos= a2_pos
    if agent_idx == 1:
        my_hold = a2_hold
        op_hold = a1_hold
        my_pos = a2_pos
        op_pos = a1_pos

    num_x, num_y = np_trgt.shape
    
    def is_valid_grid(grid):
        x, y = grid
        if x < 0 or x >= num_x:
            return False
        if y < 0 or y >= num_y:
            return False
        return True

    def is_empty_grid(grid):
        return is_valid_grid(grid) and np_trgt[grid] == 0
    

    dict_features["bias"] = 1

    if np.sum(np_trgt) == 0:
        dict_features[ON_BAG] = 0
        dict_features[HOLDING] = 0
        dict_features[BOTH_HOLD] = 0
        dict_features[DIST_TAR] = 0
        dict_features[DIST_AGENT] = 0
        dict_features[DIST_GOAL] = 0
        return dict_features

    if my_hold:
        action = AgentActions(action_idx)
        dict_features[ON_BAG] = 1
        if action == AgentActions.HOLD:
            dict_features[HOLDING] = 0
            dict_features[BOTH_HOLD] = 0
            dist_target, _ = get_nearest_target(np_trgt, my_pos)
            dict_features[DIST_TAR] = 1 / (dist_target + 1)
            dict_features[DIST_AGENT] = 1 / (manhattan_distance(my_pos, op_pos) + 1)
            dict_features[DIST_GOAL] = 0
        else:
            dict_features[HOLDING] = 1
            dict_features[BOTH_HOLD] = 0
            if my_hold and op_hold and my_pos == op_pos:
                dict_features[BOTH_HOLD] = 1
            dict_features[DIST_TAR] = 0
            new_pos = my_pos
            if action == AgentActions.UP:
                new_pos = (my_pos[0], my_pos[1] - 1)
                new_pos = new_pos if is_empty_grid(new_pos) else my_pos
            elif action == AgentActions.DOWN:
                new_pos = (my_pos[0], my_pos[1] + 1)
                new_pos = new_pos if is_empty_grid(new_pos) else my_pos
            elif action == AgentActions.LEFT:
                new_pos = (my_pos[0] - 1, my_pos[1])
                new_pos = new_pos if is_empty_grid(new_pos) else my_pos
            elif action == AgentActions.RIGHT:
                new_pos = (my_pos[0] + 1, my_pos[1])
                new_pos = new_pos if is_empty_grid(new_pos) else my_pos
            dict_features[DIST_AGENT] = 1 / (manhattan_distance(new_pos, op_pos) + 1)
            np_trgt_new = np.array(np_trgt)
            np_trgt_new[my_pos] = 0
            dict_features[DIST_GOAL] = 1 / (get_astar_distance(
                np_trgt_new, new_pos, goals, hueristic=manhattan_distance)[0] + 1)
    else:
        action = AgentActions(action_idx)
        if action == AgentActions.HOLD:
            if np_trgt[my_pos] == 1:
                dict_features[ON_BAG] = 1
                dict_features[DIST_TAR] = 0
                dict_features[DIST_GOAL] = 1 / (get_astar_distance(
                    np_trgt, my_pos, goals, hueristic=manhattan_distance)[0] + 1)
                dict_features[HOLDING] = 1
                dict_features[BOTH_HOLD] = 0
                if op_hold and my_pos == op_pos:
                    dict_features[BOTH_HOLD] = 1
            else:
                dict_features[ON_BAG] = 0
                dist_target, _ = get_nearest_target(np_trgt, my_pos)
                dict_features[DIST_TAR] = 1 / (dist_target + 1)
                dict_features[DIST_GOAL] = 0
                dict_features[HOLDING] = 0
                dict_features[BOTH_HOLD] = 0

            dict_features[DIST_AGENT] = 1 / (manhattan_distance(my_pos, op_pos) + 1)
        else:
            new_pos = my_pos
            if action == AgentActions.UP:
                new_pos = (my_pos[0], my_pos[1] - 1)
                new_pos = new_pos if is_valid_grid(new_pos) else my_pos
            elif action == AgentActions.DOWN:
                new_pos = (my_pos[0], my_pos[1] + 1)
                new_pos = new_pos if is_valid_grid(new_pos) else my_pos
            elif action == AgentActions.LEFT:
                new_pos = (my_pos[0] - 1, my_pos[1])
                new_pos = new_pos if is_valid_grid(new_pos) else my_pos
            elif action == AgentActions.RIGHT:
                new_pos = (my_pos[0] + 1, my_pos[1])
                new_pos = new_pos if is_valid_grid(new_pos) else my_pos

            dict_features[ON_BAG] = 1 if np_trgt[new_pos] == 1 else 0
            dist_target, _ = get_nearest_target(np_trgt, new_pos)
            dict_features[DIST_TAR] = 1 / (dist_target + 1)
            dict_features[DIST_GOAL] = 0
            dict_features[HOLDING] = 0
            dict_features[BOTH_HOLD] = 0
            dict_features[DIST_AGENT] = 1 / (manhattan_distance(new_pos, op_pos) + 1)

    return dict_features

def get_joint_actions_for_qlearn(state):
    return list(range(len(AgentActions)**2))

def get_indv_actions_for_qlearn(state):
    return list(range(len(AgentActions)))

# def conv_to_feature_state(state, action_idx, agent_idx, goals):
#     dict_feat = extract_qlearn_features_indv(
#         state, action_idx, agent_idx, goals)
    
#     feat_state = (
#         dict_feat[DIST_TAR], dict_feat[DIST_GOAL], dict_feat[DIST_AGENT],
#         dict_feat[ON_BAG], dict_feat[HOLDING], dict_feat[BOTH_HOLD])

def get_feature_state_indv(state, agent_idx, goals):
    np_trgt, a1_pos, a2_pos, a1_hold, a2_hold = state

    dict_features = {}
    my_hold = a1_hold
    op_hold = a2_hold
    my_pos= a1_pos
    op_pos= a2_pos
    if agent_idx == 1:
        my_hold = a2_hold
        op_hold = a1_hold
        my_pos = a2_pos
        op_pos = a1_pos

    num_x, num_y = np_trgt.shape
    
    if np.sum(np_trgt) == 0:
        # dict_features[DIST_GOAL] = 0
        # dict_features[DIST_TAR] = 0
        # dict_features[DIST_AGENT] = 0
        # dict_features[ON_BAG] = 0
        # dict_features[HOLDING] = 0
        return (0, 0, 0, 0, 0)

    if my_hold:
        dict_features[ON_BAG] = 1
        dict_features[HOLDING] = 1
        if my_hold and op_hold and my_pos == op_pos:
            dict_features[HOLDING] = 2
        dict_features[DIST_TAR] = 0
        dict_features[DIST_AGENT] = manhattan_distance(my_pos, op_pos)
        dict_features[DIST_GOAL] = get_astar_distance(
            np_trgt, my_pos, goals, hueristic=manhattan_distance)[0]
    else:
        dict_features[ON_BAG] = 1 if np_trgt[my_pos] == 1 else 0
        dict_features[HOLDING] = 0
        dist_target, _ = get_nearest_target(np_trgt, my_pos)
        dict_features[DIST_TAR] = dist_target
        dict_features[DIST_GOAL] = 0
        dict_features[DIST_AGENT] = manhattan_distance(my_pos, op_pos)

    return (
        int(dict_features[DIST_GOAL]), int(dict_features[DIST_TAR]),
        int(dict_features[DIST_AGENT]), int(dict_features[ON_BAG]),
        int(dict_features[HOLDING]))

def get_feature_state_indv_v2(state, agent_idx, goals):
    np_trgt, a1_pos, a2_pos, a1_hold, a2_hold = state

    dict_features = {}
    my_hold = a1_hold
    op_hold = a2_hold
    my_pos= a1_pos
    op_pos= a2_pos
    if agent_idx == 1:
        my_hold = a2_hold
        op_hold = a1_hold
        my_pos = a2_pos
        op_pos = a1_pos

    num_x, num_y = np_trgt.shape
    
    def is_valid_grid(grid):
        x, y = grid
        if x < 0 or x >= num_x:
            return False
        if y < 0 or y >= num_y:
            return False
        return True

    def is_empty_grid(grid):
        return is_valid_grid(grid) and np_trgt[grid] == 0
    
    if np.sum(np_trgt) == 0:
        # dict_features[DIST_GOAL] = 0
        # dict_features[DIST_TAR] = 0
        # dict_features[DIST_AGENT] = 0
        # dict_features[ON_BAG] = 0
        # dict_features[HOLDING] = 0
        # agent dir, agent dist, target dir, goal dir, on bag, hold
        return (0, 0, 0, 0, 0, 0)

    
    agent_dir_x = np.sign(op_pos[0] - my_pos[0]) + 1
    agent_dir_y = np.sign(op_pos[1] - my_pos[1]) + 1
    dict_features[DIR_AGENT] = int(3 * agent_dir_y + agent_dir_x)
    dict_features[DIST_AGENT] = 1 if manhattan_distance(my_pos, op_pos) > 1 else 0

    if my_hold:
        dict_features[ON_BAG] = 1
        dict_features[HOLDING] = 1
        if my_hold and op_hold and my_pos == op_pos:
            dict_features[HOLDING] = 2
        dict_features[DIR_TARGET] = 0
        len_path, list_path = get_astar_distance(
            np_trgt, my_pos, goals, hueristic=manhattan_distance)
        if len(list_path) != 0:
            goal_dir_x = list_path[0][0] - my_pos[0]
            goal_dir_y = list_path[0][1] - my_pos[1]
            if goal_dir_x < 0:
                dict_features[DIR_GOAL] = 4
            elif goal_dir_x > 0:
                dict_features[DIR_GOAL] = 2
            elif goal_dir_y < 0:
                dict_features[DIR_GOAL] = 1
            elif goal_dir_y > 0:
                dict_features[DIR_GOAL] = 3
            else:
                dict_features[DIR_GOAL] = 0
        else:
            dict_features[DIR_GOAL] = 0
    else:
        if np_trgt[my_pos] == 1:
            dict_features[ON_BAG] = 1
            dict_features[DIR_TARGET] = 0
        else:
            dict_features[ON_BAG] = 0
            np_dist_tar = np.zeros(4)
            # up
            new_pos = (my_pos[0], my_pos[1] - 1)
            dist_target = float("inf")
            if is_valid_grid(new_pos):
                dist_target, _ = get_nearest_target(np_trgt, new_pos)
            np_dist_tar[0] = dist_target
            # right
            new_pos = (my_pos[0] + 1, my_pos[1])
            dist_target = float("inf")
            if is_valid_grid(new_pos):
                dist_target, _ = get_nearest_target(np_trgt, new_pos)
            np_dist_tar[1] = dist_target
            # down
            new_pos = (my_pos[0], my_pos[1] + 1)
            dist_target = float("inf")
            if is_valid_grid(new_pos):
                dist_target, _ = get_nearest_target(np_trgt, new_pos)
            np_dist_tar[2] = dist_target
            # left
            new_pos = (my_pos[0] - 1, my_pos[1])
            dist_target = float("inf")
            if is_valid_grid(new_pos):
                dist_target, _ = get_nearest_target(np_trgt, new_pos)
            np_dist_tar[3] = dist_target

            list_min_idx = np.argwhere(np_dist_tar == np.min(np_dist_tar))
            dir_tar = 0
            for idx in list_min_idx:
                dir_tar += 2 ** idx

            dict_features[DIR_TARGET] = int(dir_tar)

        dict_features[HOLDING] = 0
        dict_features[DIR_GOAL] = 0

    return (
        int(dict_features[DIR_AGENT]), int(dict_features[DIST_AGENT]),
        int(dict_features[DIR_TARGET]), int(dict_features[DIR_GOAL]),
        int(dict_features[ON_BAG]), int(dict_features[HOLDING]))
