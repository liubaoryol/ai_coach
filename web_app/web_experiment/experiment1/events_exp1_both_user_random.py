from ai_coach_domain.box_push.simulator import BoxPushSimulator_AlwaysTogether
from ai_coach_domain.box_push.maps import EXP1_MAP
from ai_coach_domain.box_push.mdp import BoxPushTeamMDP_AlwaysTogether
from ai_coach_domain.box_push.mdppolicy import BoxPushPolicyTeamExp1
from ai_coach_domain.box_push.agent import BoxPushAIAgent_Team2
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl
import web_experiment.experiment1.task_data as td

EXP1_MDP = BoxPushTeamMDP_AlwaysTogether(**EXP1_MAP)
AGENT1 = BoxPushSimulator_AlwaysTogether.AGENT1
AGENT2 = BoxPushSimulator_AlwaysTogether.AGENT2

TEMPERATURE = 0.3
TEAMMATE_POLICY = BoxPushPolicyTeamExp1(EXP1_MDP, TEMPERATURE, AGENT2)

for session_name in [td.SESSION_A3, td.SESSION_A4]:
  name_space = '/' + td.EXP1_PAGENAMES[session_name]

  def run_game(msg,
               session_name=session_name,
               name_space=name_space,
               task_type=td.EXP1_TASK_TYPES[session_name]):
    agent2 = BoxPushAIAgent_Team2(TEAMMATE_POLICY)
    event_impl.run_task_game(msg, td.map_g_id_2_game[session_name], agent2,
                             None, EXP1_MAP, event_impl.ASK_LATENT, name_space,
                             task_type)

  def action_event(msg,
                   session_name=session_name,
                   name_space=name_space,
                   task_type=td.EXP1_TASK_TYPES[session_name]):
    ASK_LATENT_FREQUENCY = 5
    event_impl.action_event(msg, td.map_g_id_2_game[session_name], None,
                            event_impl.game_end, name_space, True, True,
                            ASK_LATENT_FREQUENCY, EXP1_MAP, session_name,
                            task_type)

  socketio.on_event('run_game', run_game, name_space)
  socketio.on_event('action_event', action_event, name_space)
