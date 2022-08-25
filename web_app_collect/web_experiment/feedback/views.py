from flask import redirect, render_template, request, session, url_for, g
from . import feedback_bp
import web_experiment.experiment1.define as td
from web_experiment.auth.util import load_session_trajectory, get_domain_type
from web_experiment.models import User, db
from web_experiment.define import EDomainType
from web_experiment.feedback.define import COLLECT_NAMESPACES


def get_record_session_name(game_session_name):
  return f"{game_session_name}_record"


COLLECT_TEMPLATE = {
    EDomainType.Movers: 'collect_latent_movers.html',
    EDomainType.Cleanup: 'collect_latent_cleanup.html',
}

COLLECT_NEXT_ENDPOINT = {
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
    return redirect(url_for(COLLECT_NEXT_ENDPOINT[session_record_name]))

  disabled = ''
  if not getattr(query_data, session_record_name):
    disabled = 'disabled'

  load_session_trajectory(session_name, g.user)
  lstates = [
      f"{latent_state[0]}, {latent_state[1]}"
      for latent_state in session['possible_latent_states']
  ]
  print(session['max_index'])
  domain_type = get_domain_type(session_name)
  loaded_session_title = td.EXP1_SESSION_TITLE[session_name]
  return render_template(COLLECT_TEMPLATE[domain_type],
                         cur_user=g.user,
                         is_disabled=disabled,
                         session_title=loaded_session_title,
                         session_length=session['max_index'],
                         socket_name_space=COLLECT_NAMESPACES[domain_type],
                         latent_states=lstates)
