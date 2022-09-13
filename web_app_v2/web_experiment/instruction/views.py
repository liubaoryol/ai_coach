import logging
from flask import render_template, g, request, redirect, session
from web_experiment.define import get_next_url, PageKey
from web_experiment.auth.functions import login_required
from . import inst_bp


def overview():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Overview
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to overview.' % (cur_user, ))

  return render_template('overview.html', cur_endpoint=cur_endpoint)


def movers_and_packers():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Movers_and_packers
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to movers_and_packers.' % (cur_user, ))

  return render_template('movers_and_packers.html', cur_endpoint=cur_endpoint)


def clean_up():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Clean_up
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to clean_up.' % (cur_user, ))

  return render_template('clean_up.html', cur_endpoint=cur_endpoint)


def rescue():
  cur_user = g.user
  group_id = session["groupid"]
  exp_type = session["exp_type"]
  cur_endpoint = inst_bp.name + "." + PageKey.Rescue
  if request.method == "POST":
    return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

  logging.info('User %s accesses to rescue.' % (cur_user, ))

  return render_template('rescue.html', cur_endpoint=cur_endpoint)


inst_bp.add_url_rule("/" + PageKey.Overview,
                     PageKey.Overview,
                     login_required(overview),
                     methods=("GET", "POST"))
inst_bp.add_url_rule("/" + PageKey.Movers_and_packers,
                     PageKey.Movers_and_packers,
                     login_required(movers_and_packers),
                     methods=("GET", "POST"))
inst_bp.add_url_rule("/" + PageKey.Clean_up,
                     PageKey.Clean_up,
                     login_required(clean_up),
                     methods=("GET", "POST"))
inst_bp.add_url_rule("/" + PageKey.Rescue,
                     PageKey.Rescue,
                     login_required(rescue),
                     methods=("GET", "POST"))
