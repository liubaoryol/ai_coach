import os
import time
import logging
from flask import (flash, g, redirect, render_template, request, url_for,
                   current_app, session)
from web_experiment.auth.functions import login_required
from web_experiment.models import (db, User, InExperiment, PreExperiment,
                                   PostExperiment)
from . import survey_bp
from web_experiment.auth.util import load_session_trajectory


@survey_bp.route('/preexperiment', methods=('GET', 'POST'))
@login_required
def preexperiment():
  cur_user = g.user
  if request.method == 'POST':
    logging.info('User %s submitted pre-experiment survey.' % (cur_user, ))
    query_data = User.query.filter_by(userid=cur_user).first()
    if not query_data.pre_exp:
      query_data.pre_exp = True
      db.session.commit()
    return redirect(url_for('inst.movers_and_packers'))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.pre_exp:
    disabled = 'disabled'
  return render_template('preexperiment.html', is_disabled=disabled)


def inexperiment_impl(current_html_file, next_endpoint_name, session_name):
  cur_user = g.user
  query_data = User.query.filter_by(userid=cur_user).first()
  session_survey_name = f"session_{session_name}_survey"

  if request.method == 'POST':
    logging.info('User %s submitted in-experiment survey %s.' %
                 (cur_user, session_name))
    if not getattr(query_data, session_survey_name):
      setattr(query_data, session_survey_name, True)
      db.session.commit()
    return redirect(url_for(next_endpoint_name))

  disabled = ''
  if not getattr(query_data, session_survey_name):
    disabled = 'disabled'
  return render_template(current_html_file, is_disabled=disabled)


@survey_bp.route('/survey_both_tell_align', methods=('GET', 'POST'))
@login_required
def survey_both_tell_align():
  # survey for session a0
  return inexperiment_impl('survey_both_tell_align.html',
                           'exp1.exp1_both_user_random', 'a0')


@survey_bp.route('/survey_both_user_random', methods=('GET', 'POST'))
@login_required
def survey_both_user_random():
  return inexperiment_impl('survey_both_user_random.html',
                           'exp1.exp1_both_user_random_2', 'a1')


@survey_bp.route('/survey_both_user_random_2', methods=('GET', 'POST'))
@login_required
def survey_both_user_random_2():
  return inexperiment_impl('survey_both_user_random_2.html',
                           'exp1.exp1_both_user_random_3', 'a2')


@survey_bp.route('/survey_both_user_random_3', methods=('GET', 'POST'))
@login_required
def survey_both_user_random_3():
  return inexperiment_impl('survey_both_user_random_3.html', 'inst.clean_up',
                           'a3')


@survey_bp.route('/survey_indv_tell_align', methods=('GET', 'POST'))
@login_required
def survey_indv_tell_align():
  return inexperiment_impl('survey_indv_tell_align.html',
                           'exp1.exp1_indv_user_random', "b0")


@survey_bp.route('/survey_indv_user_random', methods=('GET', 'POST'))
@login_required
def survey_indv_user_random():
  return inexperiment_impl('survey_indv_user_random.html',
                           'exp1.exp1_indv_user_random_2', "b1")


@survey_bp.route('/survey_indv_user_random_2', methods=('GET', 'POST'))
@login_required
def survey_indv_user_random_2():
  return inexperiment_impl('survey_indv_user_random_2.html',
                           'exp1.exp1_indv_user_random_3', "b2")


@survey_bp.route('/survey_indv_user_random_3', methods=('GET', 'POST'))
@login_required
def survey_indv_user_random_3():
  return inexperiment_impl('survey_indv_user_random_3.html',
                           'survey.completion', "b3")


@survey_bp.route('/completion', methods=('GET', 'POST'))
@login_required
def completion():
  cur_user = g.user
  if request.method == 'POST':
    logging.info('User %s submitted post-experiment survey.' % (cur_user, ))

    error = None
    user = User.query.filter_by(userid=cur_user).first()

    if user is None:
      error = 'Incorrect user id'
      flash(error)
    else:
      if not user.completed:
        user.completed = True
        db.session.commit()
      return redirect(url_for('survey.thankyou'))

  query_data = User.query.filter_by(userid=cur_user).first()
  disabled = ''
  if not query_data.completed:
    disabled = 'disabled'
  return render_template('completion.html', is_disabled=disabled)


@survey_bp.route('/thankyou', methods=('GET', 'POST'))
@login_required
def thankyou():
  cur_user = g.user
  session.clear()
  logging.info('User %s completed the experiment.' % (cur_user, ))
  return render_template('thankyou.html')
