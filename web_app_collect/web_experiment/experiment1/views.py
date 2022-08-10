import logging
from flask import render_template, g, session, request, url_for, redirect
from web_experiment.auth.functions import login_required
from web_experiment.models import User
from . import exp1_bp


@exp1_bp.route('/exp1_both_tell_align', methods=('GET', 'POST'))
@login_required
def exp1_both_tell_align():
  session_name = "a0"
  file_name = "exp1_both_tell_align.html"
  next_endpoint = "survey.survey_both_tell_align"
  return exp1_both_user_random_template(session_name, file_name, next_endpoint)

@exp1_bp.route('/exp1_both_user_random', methods=('GET', 'POST'))
@login_required
def exp1_both_user_random():
  session_name = "a1"
  file_name = "exp1_both_user_random.html"
  next_endpoint = "survey.survey_both_user_random"
  return exp1_both_user_random_template(session_name, file_name, next_endpoint)

@exp1_bp.route('/exp1_both_user_random_2', methods=('GET', 'POST'))
@login_required
def exp1_both_user_random_2():
  session_name = "a2"
  file_name = "exp1_both_user_random_2.html"
  next_endpoint = "survey.survey_both_user_random_2"
  return exp1_both_user_random_template(session_name, file_name, next_endpoint)

@exp1_bp.route('/exp1_both_user_random_3', methods=('GET', 'POST'))
@login_required
def exp1_both_user_random_3():
  session_name = "a3"
  file_name = "exp1_both_user_random_3.html"
  next_endpoint = "survey.survey_both_user_random_3"
  return exp1_both_user_random_template(session_name, file_name, next_endpoint)



def exp1_both_user_random_template(session_name, file_name, next_endpoint):
  cur_user = g.user
  if request.method == "POST":
    return redirect(
        url_for('feedback.collect',
                session_name=session_name,
                next_endpoint=next_endpoint))
  logging.info('User %s accesses to session %s.' % (
      cur_user,
      session_name,
  ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not getattr(query_data, f"session_{session_name}"):
    disabled = 'disabled'
  return render_template(file_name, cur_user=cur_user, is_disabled=disabled)


@exp1_bp.route('/tutorial1', methods=('GET', 'POST'))
@login_required
def tutorial1():
  cur_user = g.user
  logging.info('User %s accesses to tutorial1.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.tutorial1:
    disabled = 'disabled'
  return render_template('tutorial1.html',
                         cur_user=cur_user,
                         is_disabled=disabled)
