import logging
from flask import flash, redirect, render_template, request, session, url_for, g, current_app
# from web_experiment.db import query_db
from . import feedback_bp
from web_experiment.auth.util import load_session_trajectory, predict_human_latent_full
from web_experiment.feedback.helper import load_latent
from web_experiment.models import User

@feedback_bp.route('/collect/<session_name>', methods=('GET', 'POST'))
def collect(session_name):
    cur_user = g.user
    disabled = ''
    query_data = User.query.filter_by(userid = cur_user).first()
    if not query_data.session_a2_record:
        disabled = 'disabled'
    print("disabled: " + disabled)
    load_session_trajectory(session_name, g.user)
    lstates = [f"{latent_state[0]}, {latent_state[1]}" for latent_state in session['possible_latent_states']]

    if session_name.startswith('a'):
        return render_template("together_collect_latent.html", cur_user = g.user, is_disabled = disabled, user_id = session['replay_id'], session_name = session['session_name'], session_length = session['max_index'], max_value = session['max_index'] - 1, latent_states = lstates)

@feedback_bp.route('/feedback/<session_name>', methods=('GET', 'POST'))
def feedback(session_name):
    if request.method == "POST":
        return redirect(url_for("exp1.exp1_both_user_random_2"))

    if session['user_group'] == "B":
        load_session_trajectory(session_name, g.user)
        lstates_full = predict_human_latent_full(session['dict'], is_movers_domain=True)
        # add a dummy for the last time frame
        lstates_full.append("None")
        session['latent_human_predicted'] = lstates_full

    elif session['user_group'] == "C":
        session['latent_human_recorded'] = load_latent(session['user_id'], session_name)

    return render_template("together_feedback_true_latent.html",
                        cur_user=g.user,
                        is_disabled=True,
                        user_id=session['replay_id'],
                        session_name=session['session_name'],
                        session_length=session['max_index'],
                        max_value=session['max_index'] - 1)
                    
