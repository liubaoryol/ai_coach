import logging
from flask import flash, redirect, render_template, request, session, url_for, g, current_app
# from web_experiment.db import query_db
from . import feedback_bp
from web_experiment.auth.util import load_session_trajectory
from web_experiment.feedback.helper import load_latent

@feedback_bp.route('/collect/<session_name>', methods=('GET', 'POST'))
def collect(session_name):
    # find a better way of only storing once
      

    session['stored'] = False
    load_session_trajectory(session_name, g.user)
    lstates = [f"{latent_state[0]}, {latent_state[1]}" for latent_state in session['possible_latent_states']]
    if session_name.startswith('a'):
        return render_template("together_collect_latent.html", cur_user = g.user, is_disabled = True, user_id = session['replay_id'], session_name = session['session_name'], session_length = session['max_index'], max_value = session['max_index'] - 1, latent_states = lstates)

@feedback_bp.route('/feedback/<session_name>', methods=('GET', 'POST'))
def feedback(session_name):
    print(session)
    session['latent_human_recorded'] = load_latent(session['user_id'], session_name)
    print(session['latent_human_recorded'])
    return render_template("together_feedback_true_latent.html",
                            cur_user=g.user,
                            is_disabled=True,
                            user_id=session['replay_id'],
                            session_name=session['session_name'],
                            session_length=session['max_index'],
                            max_value=session['max_index'] - 1)
                    
