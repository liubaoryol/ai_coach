import random
from ai_coach_domain.box_push import EventType
from ai_coach_domain.box_push.simulator import BoxPushSimulator_AlwaysTogether
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.mdp import BoxPushTeamMDP_AlwaysTogether
from ai_coach_domain.box_push.mdppolicy import BoxPushPolicyTeamExp1
from ai_coach_domain.box_push.agent import BoxPushAIAgent_Host
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl
import web_experiment.experiment1.task_data as td

EXP1_MDP = BoxPushTeamMDP_AlwaysTogether(**EXP1_MAP)
AGENT1 = BoxPushSimulator_AlwaysTogether.AGENT1
AGENT2 = BoxPushSimulator_AlwaysTogether.AGENT2

TEMPERATURE = 0.3
TEAMMATE_POLICY = BoxPushPolicyTeamExp1(EXP1_MDP, TEMPERATURE, AGENT2)

for session_name in [td.SESSION_A1, td.SESSION_A2]:
  name_space = '/' + td.EXP1_PAGENAMES[session_name]

  def make_run_game(session_name, name_space=name_space):
    def run_game(msg):
      agent2 = BoxPushAIAgent_Host(TEAMMATE_POLICY)

      def set_init_latent(game: BoxPushSimulator_AlwaysTogether):
        valid_boxes = event_impl.get_valid_box_to_pickup(game)
        if len(valid_boxes) > 0:
          box_idx = random.choice(valid_boxes)
          game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", box_idx))
          game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", box_idx))

      event_impl.run_task_game(msg, td.map_g_id_2_game[session_name], agent2,
                               set_init_latent, EXP1_MAP,
                               event_impl.NOT_ASK_LATENT, name_space,
                               td.EXP1_TASK_TYPES[session_name])

    return run_game

  def make_action_event(session_name, name_space=name_space):
    def action_event(msg):
      def hold_changed(game: BoxPushSimulator_AlwaysTogether, a1_hold_changed,
                       a2_hold_changed, a1_box, a2_box):
        if a1_hold_changed:
          if a1_box >= 0:
            game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))
            game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))
          else:
            valid_boxes = event_impl.get_valid_box_to_pickup(game)
            if len(valid_boxes) > 0:
              box_idx = random.choice(valid_boxes)
              game.event_input(AGENT1, EventType.SET_LATENT,
                               ("pickup", box_idx))
              game.event_input(AGENT2, EventType.SET_LATENT,
                               ("pickup", box_idx))

      ASK_LATENT_FREQUENCY = 5
      event_impl.action_event(msg, td.map_g_id_2_game[session_name],
                              hold_changed, event_impl.game_end, name_space,
                              False, False, ASK_LATENT_FREQUENCY, EXP1_MAP,
                              session_name, td.EXP1_TASK_TYPES[session_name])

    return action_event

  socketio.on_event('run_game', make_run_game(session_name), name_space)
  socketio.on_event('action_event', make_action_event(session_name), name_space)
