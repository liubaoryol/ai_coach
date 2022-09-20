import logging
from flask import render_template, g, request, redirect, session
from web_experiment.auth.functions import login_required
from web_experiment.models import ExpDataCollection, User
from web_experiment.define import (PageKey, DATACOL_SESSIONS, get_next_url,
                                   ExpType, url_name)
from web_experiment.exp_datacollection.define import (SESSION_TITLE,
                                                      get_socket_name)
from . import exp_dcollect_bp

EXP1_TEMPLATE = {
    PageKey.DataCol_A1: 'dcol_session_a_test.html',
    PageKey.DataCol_A2: 'dcol_session_a_test.html',
    PageKey.DataCol_A3: 'dcol_session_a_test.html',
    PageKey.DataCol_A4: 'dcol_session_a_test.html',
    PageKey.DataCol_C1: 'dcol_session_c_test.html',
    PageKey.DataCol_C2: 'dcol_session_c_test.html',
    PageKey.DataCol_C3: 'dcol_session_c_test.html',
    PageKey.DataCol_C4: 'dcol_session_c_test.html',
    PageKey.DataCol_T1: 'dcol_tutorial1.html',
    PageKey.DataCol_T3: 'dcol_tutorial3.html',
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

      logging.info('User %s accesses to %s.' % (cur_user, session_name))

      user = User.query.filter_by(userid=cur_user).first()
      query_data = ExpDataCollection.query.filter_by(
          subject_id=cur_user).first()
      disabled = ''
      if not user.test and not getattr(query_data, session_name):
        disabled = 'disabled'

      # session_name is needed when initializing UserData during 'connect' event
      # There could be a little time gap from here to UserData initialization
      # We assume the same user won't access multiple sessions simulataneously
      session['loaded_session_name'] = session_name

      socket_name = get_socket_name(session_name)
      return render_template(EXP1_TEMPLATE[session_name],
                             socket_name_space=socket_name,
                             is_disabled=disabled,
                             session_title=SESSION_TITLE[session_name],
                             cur_endpoint=cur_endpoint)

    return view_func

  func = login_required(make_view_func(session_name))
  exp_dcollect_bp.add_url_rule('/' + url_name(session_name),
                               session_name,
                               func,
                               methods=('GET', 'POST'))
