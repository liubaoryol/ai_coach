from flask import render_template, session, g, request, redirect
from web_experiment.models import db, ExpIntervention, ExpDataCollection
from web_experiment.define import (ExpType, PageKey, get_domain_type,
                                   get_next_url, get_record_session_key)
from web_experiment.auth.functions import admin_required, login_required
import web_experiment.exp_intervention.define as intv
import web_experiment.exp_datacollection.define as dcol
from web_experiment.review.util import possible_latent_states
from web_experiment.review.define import get_socket_name
from . import review_bp


def replay(session_name, user_id):
  session['loaded_session_name'] = session_name

  exp_type = session["exp_type"]
  if exp_type == ExpType.Data_collection:
    session_titles = dcol.SESSION_TITLE
  elif exp_type == ExpType.Intervention:
    session_titles = intv.SESSION_TITLE

  loaded_session_title = session_titles[session_name]
  socket_name = get_socket_name(PageKey.Replay, get_domain_type(session_name))
  return render_template("replay_base.html",
                         user_id=user_id,
                         session_title=loaded_session_title,
                         socket_name_space=socket_name)


def record(session_name, user_id):
  session['loaded_session_name'] = session_name

  exp_type = session["exp_type"]
  if exp_type == ExpType.Data_collection:
    loaded_session_title = dcol.SESSION_TITLE[session_name]
  elif exp_type == ExpType.Intervention:
    loaded_session_title = intv.SESSION_TITLE[session_name]

  domain_type = get_domain_type(session_name)
  lstates = possible_latent_states(domain_type)
  socket_name = get_socket_name(PageKey.Record, domain_type)
  return render_template("record_latent_base.html",
                         user_id=user_id,
                         session_title=loaded_session_title,
                         socket_name_space=socket_name,
                         latent_states=lstates)


def review(session_name):
  cur_user = g.user
  cur_endpoint = review_bp.name + "." + PageKey.Review
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
    return redirect(get_next_url(cur_endpoint, session_name, groupid, exp_type))

  session['loaded_session_name'] = session_name

  disabled = ''
  if not getattr(query_data, session_record_name):
    disabled = 'disabled'

  domain_type = get_domain_type(session_name)
  socket_name = get_socket_name(PageKey.Review, domain_type)

  return render_template("review_base.html",
                         domain_type=domain_type.name,
                         is_disabled=disabled,
                         session_title=loaded_session_title,
                         socket_name_space=socket_name,
                         cur_endpoint=cur_endpoint,
                         session_name=session_name)


review_bp.add_url_rule("/" + PageKey.Replay + "/<session_name>/<user_id>",
                       PageKey.Replay, admin_required(replay))
review_bp.add_url_rule("/" + PageKey.Record + "/<session_name>/<user_id>",
                       PageKey.Record, admin_required(record))
review_bp.add_url_rule("/" + PageKey.Review + "/<session_name>",
                       PageKey.Review,
                       login_required(review),
                       methods=("GET", "POST"))
