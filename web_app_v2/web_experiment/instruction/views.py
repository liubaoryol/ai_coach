import logging
from flask import render_template, g
from web_experiment.auth.functions import login_required
from . import inst_bp


@inst_bp.route('/overview', methods=('GET', 'POST'))
@login_required
def overview():
  cur_user = g.user
  logging.info('User %s accesses to overview.' % (cur_user, ))

  return render_template('overview.html')


@inst_bp.route('/movers_and_packers', methods=('GET', 'POST'))
@login_required
def movers_and_packers():
  cur_user = g.user
  logging.info('User %s accesses to movers_and_packers.' % (cur_user, ))

  return render_template('movers_and_packers.html')


@inst_bp.route('/clean_up', methods=('GET', 'POST'))
@login_required
def clean_up():
  cur_user = g.user
  logging.info('User %s accesses to clean_up.' % (cur_user, ))

  return render_template('clean_up.html')
