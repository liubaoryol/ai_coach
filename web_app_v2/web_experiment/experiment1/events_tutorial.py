from typing import Mapping, Hashable
from flask import request, session
from ai_coach_domain.box_push import (EventType, BoxState, conv_box_state_2_idx)
from ai_coach_domain.box_push.simulator import BoxPushSimulator_AlwaysTogether
from ai_coach_domain.box_push.maps import TUTORIAL_MAP
from ai_coach_domain.box_push.agent import (BoxPushSimpleAgent,
                                            BoxPushInteractiveAgent)
from web_experiment import socketio
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator_AlwaysTogether]
EXP1_TUT_NAMESPACE = '/exp1_tutorial'
SESSION_NAME = 'tutorial1'
TASK_TYPE = event_impl.TASK_A

AGENT1 = BoxPushSimulator_AlwaysTogether.AGENT1
AGENT2 = BoxPushSimulator_AlwaysTogether.AGENT2
GAME_MAP = TUTORIAL_MAP


@socketio.on('connect', namespace=EXP1_TUT_NAMESPACE)
def initial_canvas():
  event_impl.initial_canvas(SESSION_NAME, TASK_TYPE)


@socketio.on('my_echo', namespace=EXP1_TUT_NAMESPACE)
def test_message(message):
  event_impl.test_message(message)


@socketio.on('disconnect_request', namespace=EXP1_TUT_NAMESPACE)
def disconnect_request():
  event_impl.disconnect_request()


@socketio.on('my_ping', namespace=EXP1_TUT_NAMESPACE)
def ping_pong():
  event_impl.ping_pong()


@socketio.on('disconnect', namespace=EXP1_TUT_NAMESPACE)
def test_disconnect():
  event_impl.test_disconnect(g_id_2_game)


@socketio.on('run_game', namespace=EXP1_TUT_NAMESPACE)
def run_game(msg):
  env_id = request.sid

  # run a game
  if env_id not in g_id_2_game:
    g_id_2_game[env_id] = BoxPushSimulator_AlwaysTogether(env_id)

  game = g_id_2_game[env_id]
  game.init_game(**GAME_MAP)

  agent1 = BoxPushInteractiveAgent()

  ask_latent = False
  if "type" in msg:
    game_type = msg["type"]
    PICKUP_BOX = 2
    if game_type == "to_box":
      agent2 = BoxPushInteractiveAgent()
      game.set_autonomous_agent(agent1, agent2)
      game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", PICKUP_BOX))
    elif game_type == "box_pickup":
      agent2 = BoxPushSimpleAgent(AGENT2, GAME_MAP["x_grid"],
                                  GAME_MAP["y_grid"], GAME_MAP["boxes"],
                                  GAME_MAP["goals"], GAME_MAP["walls"],
                                  GAME_MAP["drops"])
      game.set_autonomous_agent(agent1, agent2)
      game.a1_pos = game.boxes[PICKUP_BOX]
      game.current_step = int(msg["score"])
      game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", PICKUP_BOX))
      game.event_input(AGENT2, EventType.SET_LATENT, ("pickup", PICKUP_BOX))
    elif game_type == "to_goal":
      agent2 = BoxPushSimpleAgent(AGENT2, GAME_MAP["x_grid"],
                                  GAME_MAP["y_grid"], GAME_MAP["boxes"],
                                  GAME_MAP["goals"], GAME_MAP["walls"],
                                  GAME_MAP["drops"])
      game.set_autonomous_agent(agent1, agent2)
      game.a1_pos = game.boxes[PICKUP_BOX]
      game.a2_pos = game.boxes[PICKUP_BOX]
      game.box_states[PICKUP_BOX] = conv_box_state_2_idx(
          (BoxState.WithBoth, None), len(game.drops))
      game.current_step = int(msg["score"])
      game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))
      game.event_input(AGENT2, EventType.SET_LATENT, ("goal", 0))
    elif game_type == "trapped_scenario":
      agent2 = BoxPushInteractiveAgent()
      game.set_autonomous_agent(agent1, agent2)
      # make scenario
      TRAP_BOX = 1
      game.box_states[TRAP_BOX] = conv_box_state_2_idx(
          (BoxState.WithBoth, None), len(game.drops))
      game.a1_pos = game.boxes[TRAP_BOX]
      game.a2_pos = game.boxes[TRAP_BOX]
      game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))
    else:  # "normal"
      agent2 = BoxPushSimpleAgent(AGENT2, GAME_MAP["x_grid"],
                                  GAME_MAP["y_grid"], GAME_MAP["boxes"],
                                  GAME_MAP["goals"], GAME_MAP["walls"],
                                  GAME_MAP["drops"])
      game.set_autonomous_agent(agent1, agent2)
      game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", 0))

      if game_type == "done_task":
        event_impl.done_task(msg, SESSION_NAME)

  else:
    agent2 = BoxPushInteractiveAgent()
    game.set_autonomous_agent(agent1, agent2)
    game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", 2))

  game_env = game.get_env_info()
  dict_update = event_impl.get_game_drawing_obj(game_env, ask_latent, [],
                                                TASK_TYPE)
  session["action_count"] = 0
  event_impl.update_html_canvas(dict_update, env_id, EXP1_TUT_NAMESPACE)


@socketio.on('action_event', namespace=EXP1_TUT_NAMESPACE)
def action_event(msg):
  auto_prompt = "auto_prompt" in msg
  prompt_on_change = True
  if "to_goal" in msg:
    prompt_on_change = False

  def go_to_next(msg, game_env):
    to_box = "to_box" in msg
    box_pickup = "box_pickup" in msg
    to_goal = "to_goal" in msg

    if to_box:
      a1_latent = game_env["a1_latent"]
      if a1_latent is None or a1_latent[0] != "pickup":
        return False

      box_coord = game_env["boxes"][a1_latent[1]]
      a1_pos = game_env["a1_pos"]
      if a1_pos[0] == box_coord[0] and a1_pos[1] == box_coord[1]:
        return True

    if box_pickup:
      num_drops = len(game_env["drops"])
      num_goals = len(game_env["goals"])
      a1_box, _ = event_impl.get_holding_box_idx(game_env["box_states"],
                                                 num_drops, num_goals)
      return a1_box >= 0

    if to_goal:
      num_drops = len(game_env["drops"])
      num_goals = len(game_env["goals"])
      a1_box, _ = event_impl.get_holding_box_idx(game_env["box_states"],
                                                 num_drops, num_goals)
      return a1_box < 0

    return False

  def game_finished(game, user_id, *args, **kwargs):
    game.reset_game()
    run_game({'user_id': user_id, 'type': 'normal'})

  ASK_LATENT_FREQUENCY = 3
  event_impl.action_event(msg, g_id_2_game, None, game_finished,
                          EXP1_TUT_NAMESPACE, prompt_on_change, auto_prompt,
                          ASK_LATENT_FREQUENCY, GAME_MAP, SESSION_NAME,
                          TASK_TYPE, go_to_next)


@socketio.on('setting_event', namespace=EXP1_TUT_NAMESPACE)
def setting_event(msg):
  def go_to_next(msg):
    next_when_set = "next_when_set" in msg
    if next_when_set:
      return msg["data"] == "Set Latent"

    return False

  event_impl.setting_event(msg, g_id_2_game, EXP1_TUT_NAMESPACE, go_to_next)
