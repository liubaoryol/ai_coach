from flask import redirect, render_template, request, session, url_for, g
from web_experiment.models import User, db
from web_experiment.auth.util import (load_session_trajectory, get_domain_type,
                                      predict_human_latent_full)
from web_experiment.define import EDomainType, GROUP_C, GROUP_D
import web_experiment.experiment1.define as exp1
from web_experiment.feedback.helper import load_latent
import web_experiment.feedback.define as fb
from . import feedback_bp


def get_record_session_name(game_session_name):
  return f"{game_session_name}_record"


COLLECT_TEMPLATE = {
    EDomainType.Movers: 'collect_latent_movers.html',
    EDomainType.Cleanup: 'collect_latent_cleanup.html',
}

FEEDBACK_NEXT_ENDPOINT = {
    exp1.SESSION_A1: 'exp1.' + exp1.EXP1_PAGENAMES[exp1.SESSION_A2],
    exp1.SESSION_B1: 'exp1.' + exp1.EXP1_PAGENAMES[exp1.SESSION_B2],
}


@feedback_bp.route('/collect/<session_name>', methods=('GET', 'POST'))
def collect(session_name):
  cur_user = g.user
  query_data = User.query.filter_by(userid=cur_user).first()
  session_record_name = get_record_session_name(session_name)
  if request.method == "POST":
    if not getattr(query_data, session_record_name):
      setattr(query_data, session_record_name, True)
      db.session.commit()
    return redirect(url_for("feedback.feedback", session_name=session_name))

  disabled = ''
  if not getattr(query_data, session_record_name):
    disabled = 'disabled'

  load_session_trajectory(session_name, g.user)
  lstates = [
      f"{latent_state[0]}, {latent_state[1]}"
      for latent_state in session['possible_latent_states']
  ]

  domain_type = get_domain_type(session_name)
  loaded_session_title = exp1.EXP1_SESSION_TITLE[session_name]
  return render_template(COLLECT_TEMPLATE[domain_type],
                         cur_user=g.user,
                         is_disabled=disabled,
                         session_title=loaded_session_title,
                         session_length=session['max_index'],
                         socket_name_space=fb.COLLECT_NAMESPACES[domain_type],
                         latent_states=lstates)


@feedback_bp.route('/feedback/<session_name>', methods=('GET', 'POST'))
def feedback(session_name):
  if request.method == "POST":
    return redirect(url_for(FEEDBACK_NEXT_ENDPOINT[session_name]))

  domain_type = get_domain_type(session_name)

  filename = ""
  groupid = session['groupid']
  if groupid == GROUP_C:
    session['latent_human_recorded'] = load_latent(session['user_id'],
                                                   session_name)
    if domain_type == EDomainType.Movers:
      filename = "feedback_collected_movers.html"
    elif domain_type == EDomainType.Cleanup:
      filename = "feedback_collected_cleanup.html"
  elif groupid == GROUP_D:
    load_session_trajectory(session_name, g.user)
    lstates_full = predict_human_latent_full(session['dict'], domain_type)
    lstates_full.append("None")
    session['latent_human_predicted'] = lstates_full
    if domain_type == EDomainType.Movers:
      filename = "feedback_predicted_movers.html"
    elif domain_type == EDomainType.Cleanup:
      filename = "feedback_predicted_cleanup.html"

  loaded_session_title = exp1.EXP1_SESSION_TITLE[session_name]
  socket_namespace = fb.FEEDBACK_NAMESPACES[domain_type]
  return render_template(filename,
                         cur_user=g.user,
                         session_title=loaded_session_title,
                         session_length=session['max_index'],
                         socket_name_space=socket_namespace)
