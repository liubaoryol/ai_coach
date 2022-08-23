import logging
from flask import render_template, g, request, url_for, redirect
from web_experiment.auth.functions import login_required
from web_experiment.models import User
import web_experiment.experiment1.task_define as td
from . import exp1_bp

EXP1_TEMPLATE = {
    td.SESSION_A0: 'exp1_session_a_practice.html',
    td.SESSION_A1: 'exp1_session_a_test.html',
    td.SESSION_A2: 'exp1_session_a_test.html',
    td.SESSION_A3: 'exp1_session_a_test.html',
    td.SESSION_B0: 'exp1_session_b_practice.html',
    td.SESSION_B1: 'exp1_session_b_test.html',
    td.SESSION_B2: 'exp1_session_b_test.html',
    td.SESSION_B3: 'exp1_session_b_test.html',
    td.TUTORIAL1: 'tutorial1.html',
    td.TUTORIAL2: 'tutorial2.html',
}

EXP1_NEXT_ENDPOINT = {
    td.SESSION_A0: 'survey.survey_both_tell_align',
    td.SESSION_A1: 'feedback.collect',
    td.SESSION_A2: 'feedback.collect',
    td.SESSION_A3: 'feedback.collect',
    td.SESSION_B0: 'survey.survey_indv_tell_align',
    td.SESSION_B1: 'feedback.collect',
    td.SESSION_B2: 'feedback.collect',
    td.SESSION_B3: 'feedback.collect',
    td.TUTORIAL1: exp1_bp.name + '.exp1_both_tell_align',
    td.TUTORIAL2: exp1_bp.name + '.exp1_indv_tell_align',
}

for session_name in td.LIST_SESSIONS:

  def make_view_func(session_name):
    def view_func():
      cur_user = g.user
      if request.method == "POST":
        return redirect(
            url_for(EXP1_NEXT_ENDPOINT[session_name],
                    session_name=session_name))

      logging.info('User %s accesses to %s.' % (cur_user, session_name))

      query_data = User.query.filter_by(userid=cur_user).first()
      disabled = ''
      if not getattr(query_data, session_name):
        disabled = 'disabled'
      return render_template(
          EXP1_TEMPLATE[session_name],
          socket_name_space=td.EXP1_PAGENAMES[session_name],
          cur_user=cur_user,
          is_disabled=disabled,
          session_title=td.EXP1_SESSION_TITLE[session_name],
          post_endpoint=url_for(exp1_bp.name + '.' +
                                td.EXP1_PAGENAMES[session_name]))

    return view_func

  func = login_required(make_view_func(session_name))
  exp1_bp.add_url_rule('/' + td.EXP1_PAGENAMES[session_name],
                       td.EXP1_PAGENAMES[session_name],
                       func,
                       methods=('GET', 'POST'))

for session_name in td.LIST_TUTORIALS:

  def make_view_func(session_name):
    def view_func():
      cur_user = g.user
      logging.info('User %s accesses to %s.' % (cur_user, session_name))

      query_data = User.query.filter_by(userid=cur_user).first()
      disabled = ''
      if not getattr(query_data, session_name):
        disabled = 'disabled'
      return render_template(EXP1_TEMPLATE[session_name],
                             socket_name_space=td.EXP1_PAGENAMES[session_name],
                             cur_user=cur_user,
                             is_disabled=disabled,
                             session_title=td.EXP1_SESSION_TITLE[session_name],
                             next_endpoint=url_for(
                                 EXP1_NEXT_ENDPOINT[session_name]))

    return view_func

  func = login_required(make_view_func(session_name))
  exp1_bp.add_url_rule('/' + td.EXP1_PAGENAMES[session_name],
                       td.EXP1_PAGENAMES[session_name], func)
