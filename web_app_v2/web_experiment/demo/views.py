from flask import render_template, request, redirect, url_for, flash
from web_experiment.demo.define import SESSION_TITLE, E_SessionName
from . import demo_bp

DEMO_TEMPLATE = {
    E_SessionName.Movers_full_dcol: 'session_a_test.html',
    E_SessionName.Movers_partial_dcol: 'session_a_test.html',
    E_SessionName.Cleanup_full_dcol: 'session_b_test.html',
    E_SessionName.Cleanup_partial_dcol: 'session_b_test.html',
    E_SessionName.Rescue_full_dcol: 'session_b_test.html',
    E_SessionName.Rescue_partial_dcol: 'session_b_test.html',
    E_SessionName.Movers_partial_normal: 'session_a_normal.html'
}

for e_session in E_SessionName:

  def make_view_func(e_session: E_SessionName):

    def view_func():
      if request.method == "POST":
        return redirect(url_for("demo.demo"))

      return render_template(DEMO_TEMPLATE[e_session],
                             socket_name_space=e_session.name,
                             session_title=SESSION_TITLE[e_session])

    return view_func

  func = make_view_func(e_session)
  demo_bp.add_url_rule('/' + e_session.name,
                       e_session.name,
                       func,
                       methods=('GET', 'POST'))


@demo_bp.route('/demo', methods=('GET', 'POST'))
def demo():
  if request.method == 'POST':
    if 'task_type' in request.form:
      task_type = request.form['task_type']

      return redirect(url_for("demo." + task_type))
    else:
      error = "Invalid Task"
      flash(error)

  # session title
  task_types = []
  for e_session in E_SessionName:
    task_types.append(e_session.name)

  return render_template('task_selection.html', task_types=task_types)
