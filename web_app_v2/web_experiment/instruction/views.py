import logging
from flask import render_template, g, request, redirect, session
from web_experiment.define import get_next_url, PageKey, url_name, ExpType
from web_experiment.auth.functions import login_required
from . import inst_bp


def get_template_name(exp_type, keyword):
  if exp_type == ExpType.Data_collection:
    return f"dcol_{keyword}.html"
  elif exp_type == ExpType.Intervention:
    return f"intv_{keyword}.html"
  else:
    raise ValueError(f'Invalid exp_type: {exp_type}')


def overview():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Overview
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to overview.' % (cur_user, ))

  return render_template(get_template_name(exp_type, "overview"),
                         cur_endpoint=cur_endpoint)


def movers_and_packers():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Movers_and_packers
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to movers_and_packers.' % (cur_user, ))

  return render_template(get_template_name(exp_type, "movers_and_packers"),
                         cur_endpoint=cur_endpoint)


def clean_up():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Clean_up
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to clean_up.' % (cur_user, ))

  return render_template(get_template_name(exp_type, "clean_up"),
                         cur_endpoint=cur_endpoint)


def rescue():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Rescue
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to rescue.' % (cur_user, ))

  return render_template(get_template_name(exp_type, "rescue"),
                         cur_endpoint=cur_endpoint)


def description_review():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Description_Review
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to the description of review.' % (cur_user, ))

  return render_template('description_review.html', cur_endpoint=cur_endpoint)


def description_select_destination():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Description_Select_Destination
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to the description of select destination.' %
               (cur_user, ))

  return render_template('description_select_destination.html',
                         cur_endpoint=cur_endpoint)


inst_bp.add_url_rule("/" + url_name(PageKey.Overview),
                     PageKey.Overview,
                     login_required(overview),
                     methods=("GET", "POST"))
inst_bp.add_url_rule("/" + url_name(PageKey.Movers_and_packers),
                     PageKey.Movers_and_packers,
                     login_required(movers_and_packers),
                     methods=("GET", "POST"))
inst_bp.add_url_rule("/" + url_name(PageKey.Clean_up),
                     PageKey.Clean_up,
                     login_required(clean_up),
                     methods=("GET", "POST"))
inst_bp.add_url_rule("/" + url_name(PageKey.Rescue),
                     PageKey.Rescue,
                     login_required(rescue),
                     methods=("GET", "POST"))
inst_bp.add_url_rule("/" + url_name(PageKey.Description_Review),
                     PageKey.Description_Review,
                     login_required(description_review),
                     methods=("GET", "POST"))
inst_bp.add_url_rule("/" + url_name(PageKey.Description_Select_Destination),
                     PageKey.Description_Select_Destination,
                     login_required(description_select_destination),
                     methods=("GET", "POST"))
