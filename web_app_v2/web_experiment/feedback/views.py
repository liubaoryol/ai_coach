from flask import redirect, render_template, request, session, g
from web_experiment.models import db, ExpIntervention, ExpDataCollection
from web_experiment.define import (ExpType, get_next_url, PageKey,
                                   get_record_session_key, get_domain_type)
from web_experiment.auth.functions import login_required
from web_experiment.auth.util import (load_session_trajectory,
                                      predict_human_latent_full)
from web_experiment.define import EDomainType, GroupName
import web_experiment.exp_intervention.define as intv
import web_experiment.exp_datacollection.define as dcol
from web_experiment.feedback.helper import load_latent
import web_experiment.feedback.define as fb
from . import feedback_bp

COLLECT_TEMPLATE = {
    EDomainType.Movers: 'collect_latent_movers.html',
    EDomainType.Cleanup: 'collect_latent_cleanup.html',
}

# FEEDBACK_NEXT_ENDPOINT = {
#     expdef.SESSION_A1: 'expdef.' + expdef.EXP1_PAGENAMES[expdef.SESSION_A2],
#     expdef.SESSION_B1: 'expdef.' + expdef.EXP1_PAGENAMES[expdef.SESSION_B2],
# }


def collect(session_name):
  cur_user = g.user
  cur_endpoint = feedback_bp.name + "." + PageKey.Collect
  group_id = session["groupid"]
  exp_type = session["exp_type"]

  if exp_type == ExpType.Data_collection:
    query_data = ExpDataCollection.query.filter_by(subject_id=cur_user).first()
    loaded_session_title = dcol.SESSION_TITLE[session_name]
  elif exp_type == ExpType.Intervention:
    query_data = ExpIntervention.query.filter_by(subject_id=cur_user).first()
    loaded_session_title = intv.SESSION_TITLE[session_name]

  session_record_name = get_record_session_key(session_name)
  if request.method == "POST":
    if not getattr(query_data, session_record_name):
      setattr(query_data, session_record_name, True)
      db.session.commit()
    return redirect(get_next_url(cur_endpoint, session_name, group_id,
                                 exp_type))
    # return redirect(url_for("feedback.feedback", session_name=session_name))

  disabled = ''
  if not getattr(query_data, session_record_name):
    disabled = 'disabled'

  load_session_trajectory(session_name, g.user)
  lstates = [
      f"{latent_state[0]}, {latent_state[1]}"
      for latent_state in session['possible_latent_states']
  ]

  domain_type = get_domain_type(session_name)
  return render_template(COLLECT_TEMPLATE[domain_type],
                         cur_user=g.user,
                         is_disabled=disabled,
                         session_title=loaded_session_title,
                         session_length=session['max_index'],
                         socket_name_space=fb.COLLECT_NAMESPACES[domain_type],
                         latent_states=lstates)


def feedback(session_name):
  cur_endpoint = feedback_bp.name + "." + PageKey.Feedback
  groupid = session["groupid"]
  exp_type = session["exp_type"]
  if request.method == "POST":
    # return redirect(url_for(FEEDBACK_NEXT_ENDPOINT[session_name]))
    return redirect(get_next_url(cur_endpoint, session_name, groupid, exp_type))

  domain_type = get_domain_type(session_name)

  filename = ""
  if groupid == GroupName.Group_C:
    session['latent_human_recorded'] = load_latent(session['user_id'],
                                                   session_name)
    if domain_type == EDomainType.Movers:
      filename = "feedback_collected_movers.html"
    elif domain_type == EDomainType.Cleanup:
      filename = "feedback_collected_cleanup.html"
  elif groupid == GroupName.Group_D:
    load_session_trajectory(session_name, g.user)
    lstates_full = predict_human_latent_full(session['dict'], domain_type)
    lstates_full.append("None")
    session['latent_human_predicted'] = lstates_full
    if domain_type == EDomainType.Movers:
      filename = "feedback_predicted_movers.html"
    elif domain_type == EDomainType.Cleanup:
      filename = "feedback_predicted_cleanup.html"

  if exp_type == ExpType.Data_collection:
    loaded_session_title = dcol.SESSION_TITLE[session_name]
  elif exp_type == ExpType.Intervention:
    loaded_session_title = intv.SESSION_TITLE[session_name]

  return render_template(filename,
                         cur_user=g.user,
                         session_title=loaded_session_title,
                         session_length=session['max_index'],
                         socket_name_space=fb.FEEDBACK_NAMESPACES[domain_type])


feedback_bp.add_url_rule("/" + PageKey.Collect + "/<session_name>",
                         PageKey.Collect,
                         login_required(collect),
                         methods=("GET", "POST"))
feedback_bp.add_url_rule("/" + PageKey.Feedback + "/<session_name>",
                         PageKey.Feedback,
                         login_required(feedback),
                         methods=("GET", "POST"))
