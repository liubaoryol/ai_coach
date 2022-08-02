import logging
from flask import render_template, g, session, request, url_for, redirect
from web_experiment.auth.functions import login_required
from web_experiment.models import User
from . import exp1_bp


@exp1_bp.route('/exp1_both_tell_align')
@login_required
def exp1_both_tell_align():
  cur_user = g.user
  logging.info('User %s accesses to exp1_both_tell_align.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_a0:
    disabled = 'disabled'
  return render_template('exp1_both_tell_align.html',
                         cur_user=cur_user,
                         is_disabled=disabled)


@exp1_bp.route('/exp1_both_user_random')
@login_required
def exp1_both_user_random():
  cur_user = g.user
  logging.info('User %s accesses to exp1_both_user_random.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_a1:
    disabled = 'disabled'
  return render_template('exp1_both_user_random.html',
                         cur_user=cur_user,
                         is_disabled=disabled)


@exp1_bp.route('/exp1_both_user_random_2', methods=('GET', 'POST'))
@login_required
def exp1_both_user_random_2():
  

  cur_user = g.user
  session_name = "a2"
  next_endpoint = "survey.survey_both_user_random_2"
  logging.info('User %s accesses to exp1_both_user_random_2.' % (cur_user, ))

  if request.method == "POST":
    if session['groupid'] == 'A' or session['groupid'] == 'B':
      return redirect(url_for(next_endpoint))
    elif session['groupid'] == 'C':
      return redirect(url_for('feedback.collect', session_name = session_name, next_endpoint = next_endpoint))
    elif session['groupid'] == 'D':
      return redirect(url_for('feedback.feedback', session_name = session_name, next_endpoint = next_endpoint))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  filename = "exp1_both_user_random_2.html"
  if not query_data.session_a2:
    disabled = 'disabled'
  if session["groupid"] == "B":
    filename = "exp1_both_user_random_2_intervention.html"
  return render_template(filename,
                         cur_user=cur_user,
                         is_disabled=disabled)

@exp1_bp.route('/exp1_indv_user_random')
@login_required
def exp1_indv_user_random():
  cur_user = g.user
  logging.info('User %s accesses to exp1_indv_user_random.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_b1:
    disabled = 'disabled'
  return render_template('exp1_indv_user_random.html',
                         cur_user=cur_user,
                         is_disabled=disabled)


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

@exp1_bp.route('/tutorial2', methods=('GET', 'POST'))
@login_required
def tutorial2():
  cur_user = g.user
  logging.info('User %s accesses to tutorial2.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.tutorial2:
    disabled = 'disabled'
  return render_template('tutorial2.html',
                         cur_user=cur_user,
                         is_disabled=disabled)
