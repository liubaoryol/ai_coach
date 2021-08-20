from flask import render_template
from web_experiment.auth.functions import login_required
from . import inst_bp


@inst_bp.route('/instruction', methods=('GET', 'POST'))
@login_required
def instruction():
  return render_template('instruction.html')
