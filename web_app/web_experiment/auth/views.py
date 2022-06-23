import logging
import re
from flask import (flash, redirect, render_template, request, url_for, g, current_app, session)
from web_experiment.models import (db, User, PostExperiment, InExperiment,
                                   PreExperiment)
from web_experiment.auth.functions import admin_required
from web_experiment.auth.util import read_file
from ai_coach_domain.box_push.helper import get_possible_latent_states
from . import auth_bp
import glob, os


@auth_bp.route('/register', methods=('GET', 'POST'))
@admin_required
def register():
  if request.method == 'POST':
    if 'userid' in request.form:
      userid = request.form['userid'].lower()
      error = None

      if not userid:
        error = 'User ID is required.'
      elif not bool(re.match('^[a-zA-Z0-9]*$', userid)):
        error = 'Invalid User ID.'
      else:
        qdata = User.query.filter_by(userid=userid).first()
        if qdata is not None:
          error = 'User {} is already registered.'.format(userid)

      if error is None:
        new_user = User(userid=userid)
        db.session.add(new_user)
        db.session.commit()
        logging.info('User %s added a new user %s.' % (g.user, userid))
        return redirect(url_for('auth.register'))

      flash(error)
    elif 'delid' in request.form:
      delid = request.form['delid'].lower()

      query_pre = PreExperiment.query.filter_by(subject_id=delid).first()
      if query_pre is not None:
        db.session.delete(query_pre)

      query_in = InExperiment.query.filter_by(subject_id=delid).all()
      for query in query_in:
        if query is not None:
          db.session.delete(query)

      query_post = PostExperiment.query.filter_by(subject_id=delid).first()
      if query_post is not None:
        db.session.delete(query_post)

      qdata = User.query.filter_by(userid=delid).first()
      if qdata is not None:
        db.session.delete(qdata)
      db.session.commit()

      logging.info('User %s deleted user %s.' % (g.user, delid))
      return redirect(url_for('auth.register'))
    elif 'replayid' in request.form and 'replaysession' in request.form:
      replayid = request.form['replayid'].lower()
      session_name = request.form['replaysession']

      user = User.query.filter_by(userid=replayid).first()
      error = None
      if not replayid:
        error = 'User ID is required.'
      elif not bool(re.match('^[a-zA-Z0-9]*$', replayid)):
        error = 'Invalid User ID.'
      elif not user:
        error = f"User {replayid} not found"
      else:
        traj_path = current_app.config["TRAJECTORY_PATH"]
        path = f"{replayid}/session_{session_name}_{replayid}*.txt"
        fileExpr = os.path.join(traj_path, path)
        # find any matching files
        files = glob.glob(fileExpr)
      
        if len(files) == 0:
          # does not find a match, error handling
          error = f"No file found that matches {replayid}, session {session_name}"
        else:
          file = files[0]
          traj = read_file(file)
          session["dict"] = traj
          session['index'] = 0
          session['max_index'] = len(traj)
          session['replay_id'] = replayid
          session['session_name'] = session_name
          session['possible_latent_states'] = get_possible_latent_states(len(traj[0]['boxes']), len(traj[0]['drops']), len(traj[0]['goals']))
          # dummy latent human prediction
          session['latent_human_predicted'] = ["None"] * session['max_index']
          session['latent_human_recorded'] = ["None"] * session['max_index']
          if session_name.startswith('a'):
            # return redirect(url_for('auth.replayA'))
            return redirect(url_for('replay.record', session_name = session_name))
          elif session_name.startswith('b'):
            # return redirect(url_for('auth.replayB'))
            return redirect(url_for('replay.record', session_name = session_name))

      if (error):
        flash(error)


  user_list = User.query.all()
  return render_template('register.html', userlist=user_list)

@auth_bp.route('/replayA')
@admin_required
def replayA():
  lstates = [f"{latent_state[0]}, {latent_state[1]}" for latent_state in session['possible_latent_states']]
  return render_template("replay_together_record_latent.html", cur_user = g.user, is_disabled = True, user_id = session['replay_id'], session_name = session['session_name'], session_length = session['max_index'], max_value = session['max_index'] - 1, latent_states = lstates)
  # return render_template("replay_together.html", cur_user = g.user, is_disabled = True, user_id = session['replay_id'], session_name = session['session_name'], session_length = session['max_index'], max_value = session['max_index'] - 1)

@auth_bp.route('/replayB')
@admin_required
def replayB():
  return render_template("replay_alone.html", cur_user = g.user, is_disabled = True, user_id = session['replay_id'], session_name = session['session_name'], session_length = session['max_index'], max_value = session['max_index'] - 1)