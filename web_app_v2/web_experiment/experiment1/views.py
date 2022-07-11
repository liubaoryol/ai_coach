import logging
from flask import render_template, g, session
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
  if not query_data.session_a1:
    disabled = 'disabled'
  return render_template('exp1_both_tell_align.html',
                         cur_user=cur_user,
                         is_disabled=disabled)

@exp1_bp.route('/exp1_both_tell_align_2')
@login_required
def exp1_both_tell_align_2():
  cur_user = g.user
  logging.info('User %s accesses to exp1_both_tell_align_2.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_a2:
    disabled = 'disabled'
  return render_template('exp1_both_tell_align_2.html',
                         cur_user=cur_user,
                         is_disabled=disabled)


@exp1_bp.route('/exp1_both_user_random')
@login_required
def exp1_both_user_random():
  cur_user = g.user
  logging.info('User %s accesses to exp1_both_user_random.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_a3:
    disabled = 'disabled'
  return render_template('exp1_both_user_random.html',
                         cur_user=cur_user,
                         is_disabled=disabled)


@exp1_bp.route('/exp1_both_user_random_2')
@login_required
def exp1_both_user_random_2():
  cur_user = g.user
  logging.info('User %s accesses to exp1_both_user_random_2.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  filename = "exp1_both_user_random_2.html"
  if not query_data.session_a4:
    disabled = 'disabled'
  if session["user_group"] == "A":
    filename = "exp1_both_user_random_2_intervention.html"
  print(filename)
  return render_template(filename,
                         cur_user=cur_user,
                         is_disabled=disabled)


@exp1_bp.route('/exp1_indv_tell_align')
@login_required
def exp1_indv_tell_align():
  cur_user = g.user
  logging.info('User %s accesses to exp1_indv_tell_align.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_b1:
    disabled = 'disabled'
  return render_template('exp1_indv_tell_align.html',
                         cur_user=cur_user,
                         is_disabled=disabled)


@exp1_bp.route('/exp1_indv_tell_random')
@login_required
def exp1_indv_tell_random():
  cur_user = g.user
  logging.info('User %s accesses to exp1_indv_tell_align.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_b2:
    disabled = 'disabled'
  return render_template('exp1_indv_tell_random.html',
                         cur_user=cur_user,
                         is_disabled=disabled)


@exp1_bp.route('/exp1_indv_user_random')
@login_required
def exp1_indv_user_random():
  cur_user = g.user
  logging.info('User %s accesses to exp1_indv_user_random.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_b3:
    disabled = 'disabled'
  return render_template('exp1_indv_user_random.html',
                         cur_user=cur_user,
                         is_disabled=disabled)


@exp1_bp.route('/exp1_indv_user_random_2')
@login_required
def exp1_indv_user_random_2():
  cur_user = g.user
  logging.info('User %s accesses to exp1_indv_user_random_2.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_b4:
    disabled = 'disabled'
  return render_template('exp1_indv_user_random_2.html',
                         cur_user=cur_user,
                         is_disabled=disabled)


@exp1_bp.route('/exp1_indv_user_random_3')
@login_required
def exp1_indv_user_random_3():
  cur_user = g.user
  logging.info('User %s accesses to exp1_indv_user_random_3.' % (cur_user, ))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.session_b5:
    disabled = 'disabled'
  return render_template('exp1_indv_user_random_3.html',
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
