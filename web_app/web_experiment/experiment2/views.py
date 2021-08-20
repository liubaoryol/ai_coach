from flask import render_template, g
from web_experiment.auth.functions import login_required
from . import exp2_bp


@exp2_bp.route('/experiment2')
@login_required
def experiment():
  cur_user = g.user
  print(cur_user)
  return render_template('experiment2.html', cur_user=cur_user)
