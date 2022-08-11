import logging
from flask import flash, redirect, render_template, request, session, url_for, g, current_app
# from web_experiment.db import query_db
from . import feedback_bp
from web_experiment.auth.util import load_session_trajectory, predict_human_latent_full
from web_experiment.feedback.helper import load_latent
from web_experiment.models import User, db


@feedback_bp.route('/collect/<session_name>/<next_endpoint>',
                   methods=('GET', 'POST'))
def collect(session_name, next_endpoint):
    cur_user = g.user
    disabled = ''
    query_data = User.query.filter_by(userid = cur_user).first()
    session_record_name = f"session_{session_name}_record"
    if request.method == "POST":
        if not getattr(query_data, session_record_name):
            setattr(query_data, session_record_name, True)
            db.session.commit()
        return redirect(url_for(next_endpoint))
    
    if not getattr(query_data, session_record_name):
        disabled = 'disabled'

    load_session_trajectory(session_name, g.user)
    lstates = [f"{latent_state[0]}, {latent_state[1]}" for latent_state in session['possible_latent_states']]
    print(session['max_index'])
    if session_name.startswith('a'):
        return render_template("together_collect_latent.html", cur_user = g.user, is_disabled = disabled, session_name = session_name, session_length = session['max_index'], max_value = session['max_index'], latent_states = lstates)
    elif session_name.startswith('b'):
        return render_template("indv_collect_latent.html", cur_user = g.user, is_disabled = disabled, session_name = session_name, session_length = session['max_index'], max_value = session['max_index'], latent_states = lstates)
                    
