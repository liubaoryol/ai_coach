import logging
from flask import render_template, g, session, request, redirect
from web_experiment.auth.functions import login_required
from web_experiment.models import ExpIntervention
from web_experiment.define import (PageKey, INTERV_SESSIONS, get_next_url,
                                   ExpType, url_name, GroupName)
from web_experiment.exp_intervention.define import (SESSION_TITLE,
                                                    get_socket_name)
from . import exp_interv_bp

EXP1_TEMPLATE = {
    # PageKey.Interv_A0: 'intv_session_a_practice.html',
    PageKey.Interv_A1: 'intv_session_a_test.html',
    PageKey.Interv_A2: 'intv_session_a_test.html',
    PageKey.Interv_A3: 'intv_session_a_test.html',
    PageKey.Interv_A4: 'intv_session_a_test.html',
    # PageKey.Interv_C0: 'intv_session_c_practice.html',
    PageKey.Interv_C1: 'intv_session_c_test.html',
    PageKey.Interv_C2: 'intv_session_c_test.html',
    PageKey.Interv_C3: 'intv_session_c_test.html',
    PageKey.Interv_C4: 'intv_session_c_test.html',
    PageKey.Interv_T1: 'intv_tutorial1.html',
    PageKey.Interv_T3: 'intv_tutorial3.html',
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
      hidden = ''
      if not getattr(query_data, session_name):
        disabled = 'disabled'
      if group_id != GroupName.Group_B:
        hidden = 'hidden'

      # session_name is needed when initializing UserData during 'connect' event
      # There could be a little time gap from here to UserData initialization
      # We assume the same user won't access multiple sessions simulataneously
      session['loaded_session_name'] = session_name

      socket_name = get_socket_name(session_name, group_id)
      return render_template(EXP1_TEMPLATE[session_name],
                             socket_name_space=socket_name,
                             cur_endpoint=cur_endpoint,
                             is_disabled=disabled,
                             is_hidden=hidden,
                             session_title=SESSION_TITLE[session_name])

    return login_required(view)

  func = make_view(session_name)
  exp_interv_bp.add_url_rule('/' + url_name(session_name),
                             session_name,
                             func,
                             methods=('GET', 'POST'))
