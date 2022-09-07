import logging
from flask import render_template, g, request, redirect, session
from web_experiment.auth.functions import login_required
from web_experiment.models import ExpDataCollection
from web_experiment.define import (PageKey, DATACOL_SESSIONS, get_next_url,
                                   ExpType)
from web_experiment.exp_datacollection.define import (SESSION_TITLE,
                                                      get_socket_name)
from . import exp_dcollect_bp

EXP1_TEMPLATE = {
    PageKey.DataCol_A0: 'exp1_session_a_practice.html',
    PageKey.DataCol_A1: 'exp1_session_a_test.html',
    PageKey.DataCol_A2: 'exp1_session_a_test.html',
    PageKey.DataCol_A3: 'exp1_session_a_test.html',
    PageKey.DataCol_B0: 'exp1_session_b_practice.html',
    PageKey.DataCol_B1: 'exp1_session_b_test.html',
    PageKey.DataCol_B2: 'exp1_session_b_test.html',
    PageKey.DataCol_B3: 'exp1_session_b_test.html',
    PageKey.DataCol_C0: 'exp1_session_c_practice.html',
    PageKey.DataCol_C1: 'exp1_session_c_test.html',
    PageKey.DataCol_C2: 'exp1_session_c_test.html',
    PageKey.DataCol_C3: 'exp1_session_c_test.html',
    PageKey.DataCol_T1: 'tutorial1.html',
    PageKey.DataCol_T2: 'tutorial2.html',
    PageKey.DataCol_T3: 'tutorial3.html',
}

for session_name in DATACOL_SESSIONS:

  def make_view_func(session_name):
    def view_func():
      cur_endpoint = exp_dcollect_bp.name + "." + session_name
      cur_user = g.user
      if request.method == "POST":
        return redirect(
            get_next_url(cur_endpoint, session_name, session['groupid'],
                         ExpType.Data_collection))
        # url_for(get_next_endpoint(cur_endpoint, session_name),
        #         session_name=session_name))

      logging.info('User %s accesses to %s.' % (cur_user, session_name))

      query_data = ExpDataCollection.query.filter_by(
          subject_id=cur_user).first()
      disabled = ''
      if not getattr(query_data, session_name):
        disabled = 'disabled'

      # session_name is needed when initializing UserData during 'connect' event
      # There could be a little time gap from here to UserData initialization
      # We assume the same user won't access multiple sessions simulataneously
      session['loaded_session_name'] = session_name

      socket_name = get_socket_name(session_name)
      return render_template(EXP1_TEMPLATE[session_name],
                             socket_name_space=socket_name,
                             cur_user=cur_user,
                             is_disabled=disabled,
                             session_title=SESSION_TITLE[session_name])

    return view_func

  func = login_required(make_view_func(session_name))
  exp_dcollect_bp.add_url_rule('/' + session_name,
                               session_name,
                               func,
                               methods=('GET', 'POST'))
