from flask import redirect, render_template, request, session, url_for, g
from . import feedback_bp
import web_experiment.experiment1.task_define as td
from web_experiment.auth.util import load_session_trajectory, get_domain_type
from web_experiment.models import User, db
from web_experiment.define import EDomainType
from web_experiment.feedback.define import (COLLECT_NAMESPACES,
                                            SESSION_COLLECT_A,
                                            SESSION_COLLECT_B)


def get_record_session_name(game_session_name):
  return f"{game_session_name}_record"


def get_collect_session_namepsace(game_session_name):
  domain_type = get_domain_type(game_session_name)
  if domain_type == EDomainType.Movers:
    return COLLECT_NAMESPACES[SESSION_COLLECT_A]
  elif domain_type == EDomainType.Cleanup:
    return COLLECT_NAMESPACES[SESSION_COLLECT_B]
  else:
    raise NotImplementedError


RECORD_NEXT_ENDPOINT = {
    get_record_session_name(td.SESSION_A0): 'survey.survey_both_tell_align',
    get_record_session_name(td.SESSION_A1): 'survey.survey_both_user_random',
    get_record_session_name(td.SESSION_A2): 'survey.survey_both_user_random_2',
    get_record_session_name(td.SESSION_A3): 'survey.survey_both_user_random_3',
    get_record_session_name(td.SESSION_B0): 'survey.survey_indv_tell_align',
    get_record_session_name(td.SESSION_B1): 'survey.survey_indv_user_random',
    get_record_session_name(td.SESSION_B2): 'survey.survey_indv_user_random_2',
    get_record_session_name(td.SESSION_B3): 'survey.survey_indv_user_random_3',
}


@feedback_bp.route('/collect/<session_name>', methods=('GET', 'POST'))
def collect(session_name):
  cur_user = g.user
  query_data = User.query.filter_by(userid=cur_user).first()
  session_record_name = get_record_session_name(session_name)
  if request.method == "POST":
    if not getattr(query_data, session_record_name):
      setattr(query_data, session_record_name, True)
      db.session.commit()
    return redirect(url_for(RECORD_NEXT_ENDPOINT[session_record_name]))

  disabled = ''
  if not getattr(query_data, session_record_name):
    disabled = 'disabled'

  load_session_trajectory(session_name, g.user)
  lstates = [
      f"{latent_state[0]}, {latent_state[1]}"
      for latent_state in session['possible_latent_states']
  ]
  print(session['max_index'])
  loaded_session_title = td.EXP1_SESSION_TITLE[session_name]
  socket_namespace = get_collect_session_namepsace(session_name)
  return render_template("collect_latent_base.html",
                         cur_user=g.user,
                         is_disabled=disabled,
                         session_title=loaded_session_title,
                         session_length=session['max_index'],
                         socket_name_space=socket_namespace,
                         latent_states=lstates)
