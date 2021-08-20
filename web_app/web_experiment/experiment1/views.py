from flask import render_template, g
from web_experiment.auth.functions import login_required
from . import exp1_bp


@exp1_bp.route('/experiment1')
@login_required
def experiment():
  cur_user = g.user
  print(cur_user)
  return render_template('experiment1.html', cur_user=cur_user)
