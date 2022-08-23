from . import replay_bp
from flask import render_template, g, session
from web_experiment.define import EDomainType
from web_experiment.auth.util import get_domain_type
import web_experiment.experiment1.task_define as td
from web_experiment.replay.define import (RECORD_NAMESPACES, SESSION_RECORD_A,
                                          SESSION_RECORD_B)


def get_record_session_namepsace(game_session_name):
  domain_type = get_domain_type(game_session_name)
  if domain_type == EDomainType.Movers:
    return RECORD_NAMESPACES[SESSION_RECORD_A]
  elif domain_type == EDomainType.Cleanup:
    return RECORD_NAMESPACES[SESSION_RECORD_B]
  else:
    raise NotImplementedError


@replay_bp.route("/record/<session_name>", methods=('GET', 'POST'))
def record(session_name):
  lstates = [
      f"{latent_state[0]}, {latent_state[1]}"
      for latent_state in session['possible_latent_states']
  ]
  loaded_session_title = td.EXP1_SESSION_TITLE[session_name]
  socket_namespace = get_record_session_namepsace(session_name)
  return render_template("replay_record_latent_base.html",
                         cur_user=g.user,
                         user_id=session['replay_id'],
                         session_title=loaded_session_title,
                         session_length=session['max_index'],
                         socket_name_space=socket_namespace,
                         latent_states=lstates)
