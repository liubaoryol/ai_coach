import logging
from flask import render_template, g, url_for
from web_experiment.auth.functions import login_required
from web_experiment.models import User
import web_experiment.experiment1.task_data as td
from . import exp1_bp

EXP1_TEMPLATE = {
    td.SESSION_A1: 'exp1_session_a_practice.html',
    td.SESSION_A2: 'exp1_session_a_practice.html',
    td.SESSION_A3: 'exp1_session_a_test.html',
    td.SESSION_A4: 'exp1_session_a_test.html',
    td.SESSION_B1: 'exp1_session_b_practice.html',
    td.SESSION_B2: 'exp1_session_b_practice.html',
    td.SESSION_B3: 'exp1_session_b_test.html',
    td.SESSION_B4: 'exp1_session_b_test.html',
    td.SESSION_B5: 'exp1_session_b_test.html',
    td.TUTORIAL1: 'tutorial1.html',
    td.TUTORIAL2: 'tutorial2.html',
}

EXP1_NEXT_ENDPOINT = {
    td.SESSION_A1: td.SURVEY_ENDPOINT[td.SESSION_A1],
    td.SESSION_A2: td.SURVEY_ENDPOINT[td.SESSION_A2],
    td.SESSION_A3: td.SURVEY_ENDPOINT[td.SESSION_A3],
    td.SESSION_A4: td.SURVEY_ENDPOINT[td.SESSION_A4],
    td.SESSION_B1: td.SURVEY_ENDPOINT[td.SESSION_B1],
    td.SESSION_B2: td.SURVEY_ENDPOINT[td.SESSION_B2],
    td.SESSION_B3: td.SURVEY_ENDPOINT[td.SESSION_B3],
    td.SESSION_B4: td.SURVEY_ENDPOINT[td.SESSION_B4],
    td.SESSION_B5: td.SURVEY_ENDPOINT[td.SESSION_B5],
    td.TUTORIAL1: 'exp1.' + td.EXP1_PAGENAMES[td.SESSION_A1],
    td.TUTORIAL2: 'exp1.' + td.EXP1_PAGENAMES[td.SESSION_B1],
}

for session_name in td.EXP1_PAGENAMES:

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
