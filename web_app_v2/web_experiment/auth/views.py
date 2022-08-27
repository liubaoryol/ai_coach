import logging
import re
from flask import flash, redirect, render_template, request, url_for, g, session
from web_experiment.models import (db, User, PostExperiment, InExperiment,
                                   PreExperiment, ExpDataCollection,
                                   ExpIntervention)
from web_experiment.auth.functions import admin_required
from web_experiment.auth.util import load_session_trajectory
import web_experiment.exp_intervention.define as intv
import web_experiment.exp_datacollection.define as dcol
from web_experiment.auth.define import EDomainType, REPLAY_NAMESPACES
from web_experiment.define import (GroupName, get_domain_type, ExpType,
                                   DATACOL_TASKS, INTERV_TASKS)
from . import auth_bp


@auth_bp.route('/register', methods=('GET', 'POST'))
@admin_required
def register():
  if request.method == 'POST':
    if 'userid' in request.form:
      userid = request.form['userid'].lower()
      groupid = request.form['groupid']
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
        new_user = User(userid=userid, groupid=groupid)
        exp_dcol = ExpDataCollection(subject_id=userid)
        exp_intv = ExpIntervention(subject_id=userid)
        db.session.add(new_user)
        db.session.add(exp_dcol)
        db.session.add(exp_intv)
        db.session.commit()
        logging.info('User %s added a new user %s.' % (g.user, userid))
        return redirect(url_for('auth.register'))

      flash(error)
    elif 'delid' in request.form:
      delid = request.form['delid'].lower()

      query = ExpDataCollection.query.filter_by(subject_id=delid).first()
      if query is not None:
        db.session.delete(query)

      query = ExpIntervention.query.filter_by(subject_id=delid).first()
      if query is not None:
        db.session.delete(query)

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
            domain_type = get_domain_type(loaded_session_name)
            if domain_type == EDomainType.Movers:
              return redirect(
                  url_for(auth_bp.name + '.' + REPLAY_NAMESPACES[domain_type]))
            elif domain_type == EDomainType.Cleanup:
              return redirect(
                  url_for(auth_bp.name + '.' + REPLAY_NAMESPACES[domain_type]))
          else:
            if domain_type == EDomainType.Movers:
              return redirect(
                  url_for('replay.record', session_name=loaded_session_name))
            elif domain_type == EDomainType.Cleanup:
              return redirect(
                  url_for('replay.record', session_name=loaded_session_name))

      if (error):
        flash(error)

  user_list = []
  for user in User.query.all():
    dict_user = {
        "userid": user.userid,
        "email": user.email,
        "groupid": user.groupid,
        "completed": "Y" if user.completed else "N"
    }
    user_list.append(dict_user)

  group_ids = [
      GroupName.Group_A, GroupName.Group_B, GroupName.Group_C, GroupName.Group_D
  ]

  exp_type = session["exp_type"]
  if exp_type == ExpType.Data_collection:
    list_task_sessions = DATACOL_TASKS
    session_titles = dcol.SESSION_TITLE
  elif exp_type == ExpType.Intervention:
    list_task_sessions = INTERV_TASKS
    session_titles = intv.SESSION_TITLE

  # session title
  task_session_titles = {}
  for session_name in list_task_sessions:
    task_session_titles[session_name] = session_titles[session_name]

  return render_template('register.html',
                         userlist=user_list,
                         session_titles=task_session_titles,
                         group_ids=group_ids)


for domain_type in REPLAY_NAMESPACES:
  namespace = REPLAY_NAMESPACES[domain_type]

  def make_replay_view(namespace=namespace):
    def replay_view():
      exp_type = session["exp_type"]
      if exp_type == ExpType.Data_collection:
        session_titles = dcol.SESSION_TITLE
      elif exp_type == ExpType.Intervention:
        session_titles = intv.SESSION_TITLE

      loaded_session_name = session['loaded_session_name']
      loaded_session_title = session_titles[loaded_session_name]
      return render_template("replay_base.html",
                             user_id=session["replay_id"],
                             session_title=loaded_session_title,
                             session_length=session['max_index'],
                             socket_name_space=namespace)

    return replay_view

  func = admin_required(make_replay_view())
  auth_bp.add_url_rule('/' + namespace, namespace, func)
