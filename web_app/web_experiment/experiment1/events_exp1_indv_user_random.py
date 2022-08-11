from ai_coach_domain.box_push.simulator import BoxPushSimulator_AlwaysAlone
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.mdp import (BoxPushAgentMDP_AlwaysAlone,
                                          BoxPushTeamMDP_AlwaysAlone)
from ai_coach_domain.box_push.mdppolicy import BoxPushPolicyIndvExp1
from ai_coach_domain.box_push.agent import BoxPushAIAgent_Indv2
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl
import web_experiment.experiment1.task_data as td

EXP1_MDP = BoxPushAgentMDP_AlwaysAlone(**EXP1_MAP)
EXP1_TASK_MDP = BoxPushTeamMDP_AlwaysAlone(**EXP1_MAP)
AGENT1 = BoxPushSimulator_AlwaysAlone.AGENT1
AGENT2 = BoxPushSimulator_AlwaysAlone.AGENT2

TEMPERATURE = 0.3
TEAMMATE_POLICY = BoxPushPolicyIndvExp1(EXP1_TASK_MDP, EXP1_MDP, TEMPERATURE,
                                        AGENT2)

for session_name in [td.SESSION_B3, td.SESSION_B4, td.SESSION_B5]:
  name_space = '/' + td.EXP1_PAGENAMES[session_name]

  def make_run_game(session_name, name_space=name_space):
    def run_game(msg):
      agent2 = BoxPushAIAgent_Indv2(TEAMMATE_POLICY)
      event_impl.run_task_game(msg, td.map_g_id_2_game[session_name], agent2,
                               None, EXP1_MAP, event_impl.ASK_LATENT,
                               name_space, td.EXP1_TASK_TYPES[session_name])

    return run_game

  def make_action_event(session_name, name_space=name_space):
    def action_event(msg):
      ASK_LATENT_FREQUENCY = 5
      event_impl.action_event(msg, td.map_g_id_2_game[session_name], None,
                              event_impl.game_end, name_space, True, True,
                              ASK_LATENT_FREQUENCY, EXP1_MAP, session_name,
                              td.EXP1_TASK_TYPES[session_name])

    return action_event

  socketio.on_event('run_game', make_run_game(session_name), name_space)
  socketio.on_event('action_event', make_action_event(session_name), name_space)
