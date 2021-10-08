import os
from ai_coach_domain.box_push.simulator import BoxPushSimulator
from ai_coach_domain.box_push import EventType
import numpy as np


class NavigationSimulator(BoxPushSimulator):
  def __init__(self, id, cb_transition_navi) -> None:
    super().__init__(id, cb_transition_navi)

  def is_finished(self) -> bool:
    if self.a1_pos in self.goals[0:2] and self.a2_pos in self.goals[2:4]:
      # if len(self.box_states) == 1:
      return True
    else:
      return False
    # return super().is_finished()

  def __transition(self, a1_action, a2_action):
    list_next_env = self.transition_fn(self.box_states, self.a1_pos,
                                       self.a2_pos, a1_action, a2_action,
                                       self.boxes, self.goals, self.walls,
                                       self.drops, self.x_grid, self.y_grid)

    list_prop = []
    for item in list_next_env:
      list_prop.append(item[0])

    idx_c = np.random.choice(range(len(list_next_env)), 1, p=list_prop)[0]
    _, a1_pos, a2_pos = list_next_env[idx_c]
    self.a1_pos = a1_pos
    self.a2_pos = a2_pos

    self.changed_state.append("a1_pos")
    self.changed_state.append("a2_pos")

  def take_a_step(self, map_agent_2_action) -> None:
    a1_action = None
    if BoxPushSimulator.AGENT1 in map_agent_2_action:
      a1_action = map_agent_2_action[BoxPushSimulator.AGENT1]

    a2_action = None
    if BoxPushSimulator.AGENT2 in map_agent_2_action:
      a2_action = map_agent_2_action[BoxPushSimulator.AGENT2]

    if a1_action is None and a2_action is None:
      return

    if a1_action is None:
      a1_action = EventType.STAY

    if a2_action is None:
      a2_action = EventType.STAY

    a1_lat = None
    a2_lat = None
    if self.a1_latent is None:
      a1_lat = ("NA", 0)
    else:
      a1_lat_0 = self.a1_latent[0] if self.a1_latent[0] is not None else "NA"
      a1_lat_1 = self.a1_latent[1] if self.a1_latent[1] is not None else 0
      a1_lat = (a1_lat_0, a1_lat_1)

    if self.a2_latent is None:
      a2_lat = ("NA", 0)
    else:
      a2_lat_0 = self.a2_latent[0] if self.a2_latent[0] is not None else "NA"
      a2_lat_1 = self.a2_latent[1] if self.a2_latent[1] is not None else 0
      a2_lat = (a2_lat_0, a2_lat_1)

    cur_state = [tuple(self.box_states), tuple(self.a1_pos), tuple(self.a2_pos)]

    state = [
        self.current_step, cur_state[0], cur_state[1], cur_state[2],
        a1_action.value, a2_action.value, a1_lat, a2_lat
    ]
    self.history.append(state)
    # print(a1_lat)
    # print(a2_lat)

    self.__transition(a1_action, a2_action)
    # super().take_a_step(map_agent_2_action)
    self.current_step += 1
    self.changed_state.append("current_step")

    if self.cb_get_A1_mental_state:
      next_state = [self.box_states, self.a1_pos, self.a2_pos]
      self.a1_latent = self.cb_get_A1_mental_state(cur_state, a1_action,
                                                   a2_action, self.a1_latent,
                                                   next_state)
    if self.cb_get_A2_mental_state:
      next_state = [self.box_states, self.a1_pos, self.a2_pos]
      self.a2_latent = self.cb_get_A2_mental_state(cur_state, a1_action,
                                                   a2_action, self.a2_latent,
                                                   next_state)

  def save_history(self, file_name, header):
    dir_path = os.path.dirname(file_name)
    if dir_path != '' and not os.path.exists(dir_path):
      os.makedirs(dir_path)

    with open(file_name, 'w', newline='') as txtfile:
      # sequence
      txtfile.write(header)
      txtfile.write('\n')
      txtfile.write('# cur_step, box_state, a1_pos, a2_pos, ' +
                    'a1_act, a2_act, a1_latent, a2_latent\n')

      for step, bstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in self.history:
        txtfile.write('%d; ' % (step, ))  # cur step
        # box states
        bstt = [0]
        for idx in range(len(bstt) - 1):
          txtfile.write('%d, ' % (bstt[idx], ))
        txtfile.write('%d; ' % (bstt[-1], ))

        txtfile.write('%d, %d; ' % a1pos)
        txtfile.write('%d, %d; ' % a2pos)

        txtfile.write('%d; %d; ' % (a1act, a2act))

        txtfile.write('%s, %d; ' % a1lat)
        txtfile.write('%s, %d; ' % a2lat)
        txtfile.write('\n')

      # last state
      txtfile.write('%d; ' % (self.current_step, ))  # cur step
      # box states
      # for idx in range(len(self.box_states) - 1):
      #   txtfile.write('%d, ' % (self.box_states[idx], ))
      txtfile.write('%d; ' % (0, ))

      txtfile.write('%d, %d; ' % self.a1_pos)
      txtfile.write('%d, %d; ' % self.a2_pos)
      txtfile.write('\n')