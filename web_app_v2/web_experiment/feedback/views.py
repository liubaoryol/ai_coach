from flask import redirect, render_template, request, session, g
from web_experiment.models import db, ExpIntervention, ExpDataCollection
from web_experiment.define import (ExpType, get_next_url, PageKey,
                                   get_record_session_key, get_domain_type)
from web_experiment.auth.functions import login_required
from web_experiment.review.util import (load_session_trajectory,
                                        predict_human_latent_full)
from web_experiment.define import EDomainType, GroupName
import web_experiment.exp_intervention.define as intv
import web_experiment.exp_datacollection.define as dcol
from web_experiment.feedback.helper import load_latent, store_latent_locally
from web_experiment.feedback.define import (FEEDBACK_NAMESPACES,
                                            PRACTICE_SESSION, SocketType)
from . import feedback_bp


def collect(session_name):
  cur_user = g.user
  cur_endpoint = feedback_bp.name + "." + PageKey.Collect
  groupid = session["groupid"]
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
    # save latent
    print(session['latent_human_recorded'])
    store_latent_locally(cur_user, session_name,
                         session['latent_human_recorded'])
    return redirect(get_next_url(cur_endpoint, session_name, groupid, exp_type))

  disabled = ''
  if not getattr(query_data, session_record_name):
    disabled = 'disabled'

  load_session_trajectory(session_name, g.user)
  lstates = [
      f"{latent_state[0]}, {latent_state[1]}"
      for latent_state in session['possible_latent_states']
  ]

  socket_name = None
  domain_type = get_domain_type(session_name)
  if domain_type == EDomainType.Movers:
    if session_name in PRACTICE_SESSION:
      socket_name = SocketType.Collect_Movers_practice
    else:
      socket_name = SocketType.Collect_Movers
  elif domain_type == EDomainType.Cleanup:
    if session_name in PRACTICE_SESSION:
      socket_name = SocketType.Collect_Cleanup_practice
    else:
      socket_name = SocketType.Collect_Cleanup
  else:
    raise NotImplementedError

  return render_template("collect_latent_base.html",
                         domain_type=domain_type.name,
                         is_disabled=disabled,
                         session_title=loaded_session_title,
                         max_index=session['max_index'],
                         socket_name_space=socket_name,
                         latent_states=lstates)


def feedback(session_name):
  cur_endpoint = feedback_bp.name + "." + PageKey.Feedback
  groupid = session["groupid"]
  exp_type = session["exp_type"]
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, session_name, groupid, exp_type))

  domain_type = get_domain_type(session_name)

  if groupid == GroupName.Group_C:
    session['latent_human_recorded'] = load_latent(session['user_id'],
                                                   session_name)
  elif groupid == GroupName.Group_D:
    load_session_trajectory(session_name, g.user)
    lstates_full = predict_human_latent_full(session['dict'], domain_type)
    lstates_full.append("None")
    session['latent_human_predicted'] = lstates_full

  if exp_type == ExpType.Data_collection:
    loaded_session_title = dcol.SESSION_TITLE[session_name]
  elif exp_type == ExpType.Intervention:
    loaded_session_title = intv.SESSION_TITLE[session_name]

  return render_template("feedback_base.html",
                         domain_type=domain_type.name,
                         groupid=groupid,
                         session_title=loaded_session_title,
                         max_index=session['max_index'],
                         socket_name_space=FEEDBACK_NAMESPACES[domain_type])


feedback_bp.add_url_rule("/" + PageKey.Collect + "/<session_name>",
                         PageKey.Collect,
                         login_required(collect),
                         methods=("GET", "POST"))
feedback_bp.add_url_rule("/" + PageKey.Feedback + "/<session_name>",
                         PageKey.Feedback,
                         login_required(feedback),
                         methods=("GET", "POST"))
