import logging
import re
from flask import flash, redirect, render_template, request, url_for, g, session
from web_experiment.models import (db, User, PostExperiment, InExperiment,
                                   PreExperiment)
from web_experiment.auth.functions import admin_required
from web_experiment.auth.util import load_session_trajectory, get_domain_type
import web_experiment.experiment1.define as td
from web_experiment.auth.define import (EDomainType, REPLAY_NAMESPACES,
                                        SESSION_REPLAY_A, SESSION_REPLAY_B)
from . import auth_bp


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
    elif ('replayid' in request.form) or ('recordid' in request.form):
      # id, game session name
      if 'replayid' in request.form:
        id = request.form['replayid'].lower()
        loaded_session_name = request.form['replaysession']
      else:
        id = request.form['recordid'].lower()
        loaded_session_name = request.form['recordsession']

      user = User.query.filter_by(userid=id).first()
      error = None
      if not id:
        error = 'User ID is required.'
      elif not bool(re.match('^[a-zA-Z0-9]*$', id)):
        error = 'Invalid User ID.'
      elif not user:
        error = f"User {id} not found"
      else:
        error = load_session_trajectory(loaded_session_name, id)
        if (not error):
          if 'replayid' in request.form:
            if get_domain_type(loaded_session_name) == EDomainType.Movers:
              return redirect(
                  url_for(auth_bp.name + '.' +
                          REPLAY_NAMESPACES[SESSION_REPLAY_A]))
            elif get_domain_type(loaded_session_name) == EDomainType.Cleanup:
              return redirect(
                  url_for(auth_bp.name + '.' +
                          REPLAY_NAMESPACES[SESSION_REPLAY_B]))
          else:
            if get_domain_type(loaded_session_name) == EDomainType.Movers:
              return redirect(
                  url_for('replay.record', session_name=loaded_session_name))
            elif get_domain_type(loaded_session_name) == EDomainType.Cleanup:
              return redirect(
                  url_for('replay.record', session_name=loaded_session_name))

      if (error):
        flash(error)

  # user_list = User.query.all()
  user_list = []
  for user in User.query.all():
    dict_user = {"userid": user.userid, "email": user.email}
    user_list.append(dict_user)

  # session title
  session_titles = {}
  for session_name in td.LIST_SESSIONS:
    session_titles[session_name] = td.EXP1_SESSION_TITLE[session_name]

  return render_template('register.html',
                         userlist=user_list,
                         session_titles=session_titles)


for replay_session_name in REPLAY_NAMESPACES:
  namespace = REPLAY_NAMESPACES[replay_session_name]

  def make_replay_view(namespace=namespace):
    def replay_view():
      loaded_session_name = session['loaded_session_name']
      loaded_session_title = td.EXP1_SESSION_TITLE[loaded_session_name]
      return render_template("replay_base.html",
                             user_id=session["replay_id"],
                             session_title=loaded_session_title,
                             session_length=session['max_index'],
                             socket_name_space=namespace)

    return replay_view

  func = admin_required(make_replay_view())
  auth_bp.add_url_rule('/' + namespace, namespace, func)
