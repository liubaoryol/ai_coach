import logging
from flask import render_template, g
from web_experiment.auth.functions import login_required
from web_experiment.models import User
import web_experiment.experiment1.task_data as td
from . import exp1_bp

for session_name in td.EXP1_PAGENAMES:

  def func_tmp(session_name=session_name):
    cur_user = g.user
    logging.info('User %s accesses to %s.' % (cur_user, session_name))

    query_data = User.query.filter_by(userid=cur_user).first()
    disabled = ''
    if not getattr(query_data, session_name):
      disabled = 'disabled'
    return render_template(td.EXP1_PAGENAMES[session_name] + '.html',
                           socket_name_space=td.EXP1_PAGENAMES[session_name],
                           cur_user=cur_user,
                           is_disabled=disabled)

  func = login_required(func_tmp)
  exp1_bp.add_url_rule('/' + td.EXP1_PAGENAMES[session_name],
                       td.EXP1_PAGENAMES[session_name], func)
