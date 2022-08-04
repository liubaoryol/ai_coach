from . import replay_bp
from flask import (flash, redirect, render_template, request, url_for, g, current_app, session)

@replay_bp.route("/record/<session_name>", methods = ('GET', 'POST'))
def record(session_name):
    lstates = [f"{latent_state[0]}, {latent_state[1]}" for latent_state in session['possible_latent_states']]
    if session_name.startswith('a'):
        return render_template("replay_together_record_latent.html", cur_user = g.user, is_disabled = True, user_id = session['replay_id'], session_name = session['session_name'], session_length = session['max_index'], max_value = session['max_index'] - 1, latent_states = lstates)
    elif session_name.startswith('b'):
        return render_template("replay_alone_record_latent.html", cur_user = g.user, is_disabled = True, user_id = session['replay_id'], session_name = session['session_name'], session_length = session['max_index'], max_value = session['max_index'] - 1, latent_states = lstates)
    
    
    