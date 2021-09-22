import heapq
import numpy as np


def manhattan_distance(start, end):
  xy1 = start
  xy2 = end
  return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


class PriorityQueue:
  def __init__(self, equal_fn=None):
    self.heap = []
    self.count = 0
    self.equal_fn = equal_fn if equal_fn else lambda i1, i2: i1 == i2

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
    # If item already in priority queue with higher priority,
    #  update its priority and rebuild the heap.
    # If item already in priority queue with equal or lower priority,
    #  do nothing.
    # If item not in priority queue, do the same thing as self.push.
    for index, (p, c, i) in enumerate(self.heap):
      if self.equal_fn(i, item):
        if p <= priority:
          break
        del self.heap[index]
        self.heap.append((priority, c, item))
        heapq.heapify(self.heap)
        break
    else:
      self.push(item, priority)


# A* grid world
def get_gridworld_astar_distance(np_gridworld: np.ndarray,
                                 start_pos,
                                 list_goals,
                                 hueristic_fn=None):
  'np_gridworld: 0 for empty cells and 1 for obstacle cells'

  def estimate_cost(cur, cost_so_far):
    min_dist = float("inf")
    for end in list_goals:
      dist = hueristic_fn(cur, end)
      if min_dist > dist:
        min_dist = dist

    return cost_so_far + min_dist

  num_x, num_y = np_gridworld.shape

  def get_neighbor(pos):
    xxx, yyy = pos
    list_neighbor = []
    x_new = xxx - 1
    if x_new >= 0 and x_new < num_x:
      if np_gridworld[(x_new, yyy)] == 0:
        list_neighbor.append((x_new, yyy))
    x_new = xxx + 1
    if x_new >= 0 and x_new < num_x:
      if np_gridworld[(x_new, yyy)] == 0:
        list_neighbor.append((x_new, yyy))
    y_new = yyy - 1
    if y_new >= 0 and y_new < num_y:
      if np_gridworld[(xxx, y_new)] == 0:
        list_neighbor.append((xxx, y_new))
    y_new = yyy + 1
    if y_new >= 0 and y_new < num_y:
      if np_gridworld[(xxx, y_new)] == 0:
        list_neighbor.append((xxx, y_new))

    return list_neighbor

  def equal_item(item1, item2):
    return item1[0] == item2[0]

  frontier = PriorityQueue(equal_item)
  visited = {}

  cost_sum = 0
  frontier.push((start_pos, []), estimate_cost(start_pos, cost_sum))

  final_path = []
  while not frontier.isEmpty():
    current, path = frontier.pop()
    visited[current] = 1
    if current in list_goals:
      final_path = path
      break

    cost_sum = len(path)
    for neighbor in get_neighbor(current):
      if visited.get(neighbor, 0) != 0:
        continue

      child_item = (neighbor, path + [neighbor])
      cost_est = estimate_cost(neighbor, cost_sum + 1)
      frontier.update(child_item, cost_est)

  return final_path
