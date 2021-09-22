from flask import render_template, g
from web_experiment.auth.functions import login_required
from . import exp1_bp


@exp1_bp.route('/exp1_both_tell_align')
@login_required
def exp1_both_tell_align():
  cur_user = g.user
  # print(cur_user)
  return render_template('exp1_both_tell_align.html', cur_user=cur_user)


@exp1_bp.route('/exp1_both_user_random')
@login_required
def exp1_both_user_random():
  cur_user = g.user
  # print(cur_user)
  return render_template('exp1_both_user_random.html', cur_user=cur_user)


@exp1_bp.route('/exp1_indv_tell_align')
@login_required
def exp1_indv_tell_align():
  cur_user = g.user
  # print(cur_user)
  return render_template('exp1_indv_tell_align.html', cur_user=cur_user)


@exp1_bp.route('/exp1_indv_tell_random')
@login_required
def exp1_indv_tell_random():
  cur_user = g.user
  # print(cur_user)
  return render_template('exp1_indv_tell_random.html', cur_user=cur_user)


@exp1_bp.route('/exp1_indv_user_random')
@login_required
def exp1_indv_user_random():
  cur_user = g.user
  # print(cur_user)
  return render_template('exp1_indv_user_random.html', cur_user=cur_user)


@exp1_bp.route('/tutorial', methods=('GET', 'POST'))
@login_required
def tutorial():
  return render_template('tutorial.html')


@exp1_bp.route('/tutorial2', methods=('GET', 'POST'))
@login_required
def tutorial2():
  return render_template('tutorial2.html')