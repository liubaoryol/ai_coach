from typing import Mapping, Hashable
import logging
from flask import request, session
from ai_coach_domain.box_push import (EventType, BoxState, conv_box_state_2_idx)
from ai_coach_domain.box_push import BoxPushSimulator_AlwaysAlone
from ai_coach_domain.box_push.box_push_maps import TUTORIAL_MAP
from ai_coach_domain.box_push.box_push_policy import get_simple_action
from web_experiment import socketio
from web_experiment.models import db, User
import web_experiment.experiment1.events_impl as event_impl

g_id_2_game = {}  # type: Mapping[Hashable, BoxPushSimulator_AlwaysAlone]
EXP1_TUT_NAMESPACE = '/exp1_tutorial2'
GRID_X = TUTORIAL_MAP["x_grid"]
GRID_Y = TUTORIAL_MAP["y_grid"]

AGENT1 = BoxPushSimulator_AlwaysAlone.AGENT1
AGENT2 = BoxPushSimulator_AlwaysAlone.AGENT2
GAME_MAP = TUTORIAL_MAP


@socketio.on('connect', namespace=EXP1_TUT_NAMESPACE)
def initial_canvas():
  event_impl.initial_canvas(GRID_X, GRID_Y)


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
    g_id_2_game[env_id] = BoxPushSimulator_AlwaysAlone(env_id)

  game = g_id_2_game[env_id]
  game.init_game(**GAME_MAP)

  ask_latent = False
  if "type" in msg:
    game_type = msg["type"]
    PICKUP_BOX = 2
    if game_type == "to_box":
      game.set_autonomous_agent()
      game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", PICKUP_BOX))
    elif game_type == "box_pickup":
      game.set_autonomous_agent()
      game.a1_pos = game.boxes[PICKUP_BOX]
      game.current_step = int(msg["score"])
      game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", PICKUP_BOX))
    elif game_type == "to_goal":
      game.set_autonomous_agent()
      game.a1_pos = game.boxes[PICKUP_BOX]
      game.box_states[PICKUP_BOX] = conv_box_state_2_idx(
          (BoxState.WithAgent1, None), len(game.drops))
      game.current_step = int(msg["score"])
      game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))
    elif game_type == "trapped_scenario":
      game.set_autonomous_agent()
      # make scenario
      bidx1 = 1
      game.a1_pos = game.boxes[bidx1]
      game.box_states[bidx1] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                    len(game.drops))
      bidx2 = 0
      game.box_states[bidx2] = conv_box_state_2_idx((BoxState.WithAgent2, None),
                                                    len(game.drops))
      game.a2_pos = game.boxes[bidx2]

      game.event_input(AGENT1, EventType.SET_LATENT, ("goal", 0))
    elif game_type == "auto_prompt":
      game.set_autonomous_agent(
          cb_get_A2_action=lambda **kwargs: get_simple_action(AGENT2, **kwargs))
      game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", 0))
    else:  # "normal"
      game.set_autonomous_agent(
          cb_get_A2_action=lambda **kwargs: get_simple_action(AGENT2, **kwargs))
      game.event_input(AGENT1, EventType.SET_LATENT, ("pickup", 0))
  else:
    game.set_autonomous_agent()

  dict_update = game.get_env_info()
  dict_update["wall_dir"] = GAME_MAP["wall_dir"]
  if dict_update is not None:
    session['action_count'] = 0
    event_impl.update_html_canvas(dict_update, env_id, ask_latent,
                                  EXP1_TUT_NAMESPACE)


@socketio.on('action_event', namespace=EXP1_TUT_NAMESPACE)
def action_event(msg):
  auto_prompt = "auto_prompt" in msg

  def game_finished(game, *args, **kwargs):
    game.reset_game()
    run_game({'user_id': msg["user_id"], 'type': 'normal'})

  ASK_LATENT_FREQUENCY = 3
  event_impl.action_event(msg, g_id_2_game, None, game_finished,
                          EXP1_TUT_NAMESPACE, True, auto_prompt,
                          ASK_LATENT_FREQUENCY)


@socketio.on('set_latent', namespace=EXP1_TUT_NAMESPACE)
def set_latent(msg):
  event_impl.set_latent(msg, g_id_2_game, EXP1_TUT_NAMESPACE)


@socketio.on('done_game', namespace=EXP1_TUT_NAMESPACE)
def done_game(msg):
  cur_user = msg["data"]
  logging.info("User %s completed tutorial2" % (cur_user, ))
  user = User.query.filter_by(userid=cur_user).first()

  if user is not None:
    user.tutorial2 = True
    db.session.commit()