import logging
import re
from flask import (flash, redirect, render_template, request, url_for, g, current_app, session)
from web_experiment.models import (db, User, PostExperiment, InExperiment,
                                   PreExperiment)
from web_experiment.auth.functions import admin_required
from web_experiment.auth.util import read_file
from . import auth_bp
import glob, os
import web_experiment.experiment1.events_impl as event_impl
from ai_coach_domain.box_push.maps import EXP1_MAP
from web_experiment import socketio


TEST_NAMESPACE = '/test'
GRID_X = EXP1_MAP["x_grid"]
GRID_Y = EXP1_MAP["y_grid"]

# EXP1_NAMESPACE = '/exp1_both_tell_align'
# namespace=EXP1_NAMESPACE
idx = None
dict = None

@socketio.on('connect', namespace=TEST_NAMESPACE)
def initial_canvas():
  event_impl.initial_canvas(GRID_X, GRID_Y)
  env_id = request.sid
  if 'dict' in session and 'index' in session:
    dict = session['dict'][session['index']]
    event_impl.update_html_canvas(dict, env_id, False, TEST_NAMESPACE)

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
        
        # print("session:" + session)
        path = f"{replayid}/session_{session_name}_{replayid}*.txt"
        # example: "./data/tw2020_trajectory/tim/session_a1_tim*.txt"
        
        fileExpr = os.path.join(traj_path, path)
        print(f"fileExpr: {fileExpr}")
        # find any matching files
        files = glob.glob(fileExpr)
      
        if len(files) == 0:
          # does not find a match, error handling
          error = f"No file found that matches {replayid}, session {session_name}"
        else:
          file = files[0]
          traj = read_file(file)
          lines = traj
          session["dict"] = traj
          session['lines'] = lines
          session['index'] = 0
          session['max_index'] = len(traj)
          return redirect(url_for('auth.replay'))
      if (error):
        flash(error)


  user_list = User.query.all()
  return render_template('register.html', userlist=user_list)

@auth_bp.route('/replay', methods=('GET', 'POST'))
@admin_required
def replay():
  cur_user = g.user
  print(request.method)
  if (request.method == 'POST'):
    if 'next' in request.form:
      if session['index'] < (session['max_index'] - 1):
        session['index'] += 1
    elif 'prev' in request.form:
      if session['index'] > 0:
        session['index'] -= 1
  print(session['index'])
      

  return render_template("replay.html", cur_user = cur_user)

