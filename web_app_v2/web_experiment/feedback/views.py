from flask import redirect, render_template, request, session, url_for, g
# from web_experiment.db import query_db
from . import feedback_bp
from web_experiment.auth.util import (load_session_trajectory,
                                      predict_human_latent_full)
from web_experiment.feedback.helper import load_latent
from web_experiment.models import User, db


@feedback_bp.route('/collect/<session_name>/<next_endpoint>',
                   methods=('GET', 'POST'))
def collect(session_name, next_endpoint):
  cur_user = g.user
  disabled = ''
  query_data = User.query.filter_by(userid=cur_user).first()
  session_record_name = f"session_{session_name}_record"
  if request.method == "POST":
    if not getattr(query_data, session_record_name):
      setattr(query_data, session_record_name, True)
      db.session.commit()
    return redirect(
        url_for("feedback.feedback",
                session_name=session_name,
                next_endpoint=next_endpoint))

  if not getattr(query_data, session_record_name):
    disabled = 'disabled'

  load_session_trajectory(session_name, g.user)
  lstates = [
      f"{latent_state[0]}, {latent_state[1]}"
      for latent_state in session['possible_latent_states']
  ]

  if session_name.startswith('a'):
    return render_template("together_collect_latent.html",
                           cur_user=g.user,
                           is_disabled=disabled,
                           session_name=session_name,
                           session_length=session['max_index'],
                           max_value=session['max_index'] - 1,
                           latent_states=lstates)
  elif session_name.startswith('b'):
    return render_template("indv_collect_latent.html",
                           cur_user=g.user,
                           is_disabled=disabled,
                           session_name=session_name,
                           session_length=session['max_index'],
                           max_value=session['max_index'] - 1,
                           latent_states=lstates)


@feedback_bp.route('/feedback/<session_name>/<next_endpoint>',
                   methods=('GET', 'POST'))
def feedback(session_name, next_endpoint):
  filename = ""
  if request.method == "POST":
    return redirect(url_for(next_endpoint))

  if session['groupid'] == "C":
    if session_name.startswith('a'):
      filename = "together_feedback_latent_collected.html"
    elif session_name.startswith('b'):
      filename = "indv_feedback_latent_collected.html"
    session['latent_human_recorded'] = load_latent(session['user_id'],
                                                   session_name)
  elif session['groupid'] == "D":
    load_session_trajectory(session_name, g.user)
    lstates_full = predict_human_latent_full(
        session['dict'], is_movers_domain=session_name.startswith('a'))
    lstates_full.append("None")
    session['latent_human_predicted'] = lstates_full
    if session_name.startswith('a'):
      filename = "together_feedback_latent_predicted.html"
    else:
      filename = "indv_feedback_latent_predicted.html"

  return render_template(filename,
                         cur_user=g.user,
                         is_disabled=True,
                         session_name=session['session_name'],
                         session_length=session['max_index'],
                         max_value=session['max_index'] - 1)
