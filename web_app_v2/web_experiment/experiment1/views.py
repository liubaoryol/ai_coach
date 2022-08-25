import logging
from flask import render_template, g, session, request, url_for, redirect
from web_experiment.auth.functions import login_required
from web_experiment.models import User
import web_experiment.experiment1.define as dfn
from . import exp1_bp

EXP1_TEMPLATE = {
    dfn.SESSION_A0: 'exp1_session_a_practice.html',
    dfn.SESSION_A1: 'exp1_session_a_test.html',
    dfn.SESSION_A2: 'exp1_session_a_test.html',
    dfn.SESSION_B0: 'exp1_session_b_practice.html',
    dfn.SESSION_B1: 'exp1_session_b_test.html',
    dfn.SESSION_B2: 'exp1_session_b_test.html',
    dfn.TUTORIAL1: 'tutorial1.html',
    dfn.TUTORIAL2: 'tutorial2.html',
}

EXP1_NEXT_ENDPOINT = {
    dfn.SESSION_A0: exp1_bp.name + '.' + dfn.EXP1_PAGENAMES[dfn.SESSION_A1],
    dfn.SESSION_A1: 'survey.survey_both_user_random',
    dfn.SESSION_A2: 'survey.survey_both_user_random_2',
    dfn.SESSION_B0: exp1_bp.name + '.' + dfn.EXP1_PAGENAMES[dfn.SESSION_B1],
    dfn.SESSION_B1: 'survey.survey_indv_user_random',
    dfn.SESSION_B2: 'survey.survey_indv_user_random_2',
    dfn.TUTORIAL1: exp1_bp.name + '.' + dfn.EXP1_PAGENAMES[dfn.SESSION_A0],
    dfn.TUTORIAL2: exp1_bp.name + '.' + dfn.EXP1_PAGENAMES[dfn.SESSION_B0],
}

# practice session views
for session_name in dfn.LIST_SESSIONS:

  def make_view(session_name):
    def view():
      cur_user = g.user
      logging.info('User %s accesses to %s.' % (cur_user, session_name))

      if request.method == "POST":
        return redirect(
            url_for(EXP1_NEXT_ENDPOINT[session_name],
                    session_name=session_name,
                    groupid=session["groupid"]))

      query_data = User.query.filter_by(userid=cur_user).first()
      disabled = ''
      if not getattr(query_data, session_name):
        disabled = 'disabled'
      return render_template(
          EXP1_TEMPLATE[session_name],
          socket_name_space=dfn.get_socket_namespace(session_name,
                                                     session["groupid"]),
          cur_user=cur_user,
          is_disabled=disabled,
          session_title=dfn.EXP1_SESSION_TITLE[session_name],
          post_endpoint=url_for(exp1_bp.name + '.' +
                                dfn.EXP1_PAGENAMES[session_name]))

    return login_required(view)

  func = make_view(session_name)
  exp1_bp.add_url_rule('/' + dfn.EXP1_PAGENAMES[session_name],
                       dfn.EXP1_PAGENAMES[session_name],
                       func,
                       methods=('GET', 'POST'))

  # if request.method == "POST":
  #   if session['groupid'] == 'C':
  #     return redirect(url_for('feedback.collect',
  #                             session_name=session_name))
  #   elif session['groupid'] == 'D':
  #     return redirect(
  #         url_for('feedback.feedback', session_name=session_name))
  #   else:  # session['groupid'] == 'A' or session['groupid'] == 'B':
  #     return redirect(url_for(EXP1_NEXT_ENDPOINT[session_name]))

for session_name in dfn.LIST_TUTORIALS:

  def make_view_func(session_name):
    def view_func():
      cur_user = g.user
      logging.info('User %s accesses to %s.' % (cur_user, session_name))

      query_data = User.query.filter_by(userid=cur_user).first()
      disabled = ''
      if not getattr(query_data, session_name):
        disabled = 'disabled'
      return render_template(
          EXP1_TEMPLATE[session_name],
          socket_name_space=dfn.get_socket_namespace(session_name,
                                                     session["groupid"]),
          cur_user=cur_user,
          is_disabled=disabled,
          session_title=dfn.EXP1_SESSION_TITLE[session_name],
          next_endpoint=url_for(EXP1_NEXT_ENDPOINT[session_name]))

    return view_func

  func = login_required(make_view_func(session_name))
  exp1_bp.add_url_rule('/' + dfn.EXP1_PAGENAMES[session_name],
                       dfn.EXP1_PAGENAMES[session_name], func)
