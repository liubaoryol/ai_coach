import logging
from flask import render_template, g, session, request, redirect
from web_experiment.auth.functions import login_required
from web_experiment.models import ExpIntervention
from web_experiment.define import (PageKey, INTERV_SESSIONS, get_next_url,
                                   ExpType)
from web_experiment.exp_intervention.define import (SESSION_TITLE,
                                                    get_socket_name)
from . import exp_interv_bp

EXP1_TEMPLATE = {
    PageKey.Interv_A0: 'exp1_session_a_practice.html',
    PageKey.Interv_A1: 'exp1_session_a_test.html',
    PageKey.Interv_A2: 'exp1_session_a_test.html',
    PageKey.Interv_B0: 'exp1_session_b_practice.html',
    PageKey.Interv_B1: 'exp1_session_b_test.html',
    PageKey.Interv_B2: 'exp1_session_b_test.html',
    PageKey.Interv_T1: 'tutorial1.html',
    PageKey.Interv_T2: 'tutorial2.html',
}

# practice session views
for session_name in INTERV_SESSIONS:

  def make_view(session_name):
    def view():
      cur_endpoint = exp_interv_bp.name + "." + session_name
      group_id = session["groupid"]
      cur_user = g.user
      logging.info('User %s accesses to %s.' % (cur_user, session_name))

      if request.method == "POST":
        return redirect(
            get_next_url(cur_endpoint, session_name, group_id,
                         ExpType.Intervention))

      query_data = ExpIntervention.query.filter_by(subject_id=cur_user).first()
      disabled = ''
      if not getattr(query_data, session_name):
        disabled = 'disabled'

      # session_name is needed when initializing UserData during 'connect' event
      # There could be a little time gap from here to UserData initialization
      # We assume the same user won't access multiple sessions simulataneously
      session['loaded_session_name'] = session_name

      socket_name = get_socket_name(session_name, group_id)
      return render_template(EXP1_TEMPLATE[session_name],
                             socket_name_space=socket_name,
                             cur_user=cur_user,
                             is_disabled=disabled,
                             session_title=SESSION_TITLE[session_name])

    return login_required(view)

  func = make_view(session_name)
  exp_interv_bp.add_url_rule('/' + session_name,
                             session_name,
                             func,
                             methods=('GET', 'POST'))
