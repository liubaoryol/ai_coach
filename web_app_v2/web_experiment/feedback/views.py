from flask import redirect, render_template, request, session, g
from web_experiment.models import db, ExpIntervention, ExpDataCollection
from web_experiment.define import (ExpType, get_next_url, PageKey,
                                   get_record_session_key, get_domain_type)
from web_experiment.auth.functions import login_required
from web_experiment.review.util import possible_latent_states
import web_experiment.exp_intervention.define as intv
import web_experiment.exp_datacollection.define as dcol
from web_experiment.feedback.define import FEEDBACK_NAMESPACES, get_socket_name
from . import feedback_bp


def collect(session_name):
  cur_user = g.user
  cur_endpoint = feedback_bp.name + "." + PageKey.Collect
  groupid = session["groupid"]
  exp_type = session["exp_type"]
  session['loaded_session_name'] = session_name

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
    return redirect(get_next_url(cur_endpoint, session_name, groupid, exp_type))

  disabled = ''
  if not getattr(query_data, session_record_name):
    disabled = 'disabled'

  domain_type = get_domain_type(session_name)
  socket_name = get_socket_name(session_name, domain_type)
  lstates = possible_latent_states(domain_type)

  return render_template("collect_latent_base.html",
                         domain_type=domain_type.name,
                         is_disabled=disabled,
                         session_title=loaded_session_title,
                         socket_name_space=socket_name,
                         latent_states=lstates,
                         session_name=session_name,
                         cur_endpoint=cur_endpoint)


def feedback(session_name):
  cur_endpoint = feedback_bp.name + "." + PageKey.Feedback
  groupid = session["groupid"]
  exp_type = session["exp_type"]
  session['loaded_session_name'] = session_name
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, session_name, groupid, exp_type))

  domain_type = get_domain_type(session_name)

  if exp_type == ExpType.Data_collection:
    loaded_session_title = dcol.SESSION_TITLE[session_name]
  elif exp_type == ExpType.Intervention:
    loaded_session_title = intv.SESSION_TITLE[session_name]

  return render_template("feedback_base.html",
                         domain_type=domain_type.name,
                         groupid=groupid,
                         session_title=loaded_session_title,
                         socket_name_space=FEEDBACK_NAMESPACES[domain_type],
                         session_name=session_name,
                         cur_endpoint=cur_endpoint)


feedback_bp.add_url_rule("/" + PageKey.Collect + "/<session_name>",
                         PageKey.Collect,
                         login_required(collect),
                         methods=("GET", "POST"))
feedback_bp.add_url_rule("/" + PageKey.Feedback + "/<session_name>",
                         PageKey.Feedback,
                         login_required(feedback),
                         methods=("GET", "POST"))
