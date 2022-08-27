from . import replay_bp
from flask import render_template, g, session
from web_experiment.define import ExpType, get_domain_type
import web_experiment.exp_intervention.define as intv
import web_experiment.exp_datacollection.define as dcol
from web_experiment.replay.define import RECORD_NAMESPACES


@replay_bp.route("/record/<session_name>", methods=('GET', 'POST'))
def record(session_name):
  lstates = [
      f"{latent_state[0]}, {latent_state[1]}"
      for latent_state in session['possible_latent_states']
  ]
  exp_type = session["exp_type"]
  if exp_type == ExpType.Data_collection:
    loaded_session_title = dcol.SESSION_TITLE[session_name]
  elif exp_type == ExpType.Intervention:
    loaded_session_title = intv.SESSION_TITLE[session_name]

  socket_namespace = RECORD_NAMESPACES[get_domain_type(session_name)]
  return render_template("replay_record_latent_base.html",
                         cur_user=g.user,
                         user_id=session['replay_id'],
                         session_title=loaded_session_title,
                         session_length=session['max_index'],
                         socket_name_space=socket_namespace,
                         latent_states=lstates)
