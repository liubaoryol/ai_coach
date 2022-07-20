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
  print(disabled)
  return render_template('preexperiment.html', is_disabled = disabled)


def inexperiment_impl(exp_no, current_html_file, next_endpoint_name, session_name = None):
  cur_user = g.user
  query_data = User.query.filter_by(userid=cur_user).first()

  if request.method == 'POST':
    logging.info('User %s submitted in-experiment survey #%d.' %
                 (cur_user, exp_no))
    print(exp_no)
    if exp_no == 0:
      if not query_data.tutorial1_survey:
        query_data.tutorial1_survey = True
        db.session.commit()
    elif exp_no == 1:
      if not query_data.session_a1_survey:
        query_data.session_a1_survey = True
        db.session.commit()
    elif exp_no == 2:
      if not query_data.session_a2_survey:
        query_data.session_a2_survey = True
        db.session.commit()
    if (session_name):
      return redirect(url_for(next_endpoint_name, session_name = session_name))
    return redirect(url_for(next_endpoint_name))
  
  disabled = ''
  if exp_no == 0:
    if not query_data.tutorial1_survey:
      disabled = 'disabled'
  elif exp_no == 1:
    if not query_data.session_a1_survey:
      disabled = 'disabled'
  elif exp_no == 2:
    if not query_data.session_a2_survey:
      disabled = 'disabled'
  return render_template(current_html_file, is_disabled = disabled)

@survey_bp.route('/survey_tutorial_1', methods=('GET', 'POST'))
@login_required
def survey_tutorial_1():
  # survey for session a0
  return inexperiment_impl(0, 'survey_tutorial_1.html', 'exp1.exp1_both_user_random')

# @survey_bp.route('/survey_both_tell_align', methods=('GET', 'POST'))
# @login_required
# def survey_both_tell_align():
#   return inexperiment_impl(1, 'survey_both_tell_align.html',
#                            'exp1.exp1_both_user_random')


@survey_bp.route('/survey_both_user_random', methods=('GET', 'POST'))
@login_required
def survey_both_user_random():
  return inexperiment_impl(1, 'survey_both_user_random.html',
                           'exp1.exp1_both_user_random_2')

@survey_bp.route('/survey_both_user_random_2', methods=('GET', 'POST'))
@login_required
def survey_both_user_random_2():
  return inexperiment_impl(2, 'survey_both_user_random_2.html',
                           'survey.completion')


@survey_bp.route('/completion', methods=('GET', 'POST'))
@login_required
def completion():
  cur_user = g.user
  if request.method == 'POST':
    logging.info('User %s submitted post-experiment survey.' % (cur_user, ))
    comment = request.form['comment']
    question = request.form['question']
    email = request.form['email']

    survey_dir = os.path.join(current_app.config["SURVEY_PATH"], cur_user)
    # save somewhere
    if not os.path.exists(survey_dir):
      os.makedirs(survey_dir)

    sec, msec = divmod(time.time() * 1000, 1000)
    time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                            time.gmtime(sec)), msec)
    file_name = ('postsurvey_' + str(cur_user) + '_' + time_stamp + '.txt')
    with open(os.path.join(survey_dir, file_name), 'w', newline='') as txtfile:
      txtfile.write('id: ' + cur_user + '\n')
      txtfile.write('email: ' + email + '\n')
      txtfile.write('comment: ' + comment + '\n')
      txtfile.write('question: ' + question + '\n')
    error = None
    user = User.query.filter_by(userid=cur_user).first()

    if user is None:
      error = 'Incorrect user id'

    if error is None:
      if email is not None:
        user.email = email
        db.session.commit()

      qdata = PostExperiment.query.filter_by(subject_id=cur_user).first()
      if qdata is not None:
        db.session.delete(qdata)
        db.session.commit()

      new_post_exp = PostExperiment(comment=comment,
                                    question=question,
                                    subject_id=cur_user)
      db.session.add(new_post_exp)

      user.completed = True
      db.session.commit()

      return redirect(url_for('survey.thankyou'))

    flash(error)

  query_data = User.query.filter_by(userid=cur_user).first()
  email_db = query_data.email
  query_comments = PostExperiment.query.filter_by(subject_id=cur_user).first()

  comments = ''
  questions = ''
  if query_comments is not None:
    comments = query_comments.comment
    questions = query_comments.question

  return render_template('completion.html',
                         saved_email=email_db,
                         saved_comment=comments,
                         saved_question=questions)


@survey_bp.route('/thankyou', methods=('GET', 'POST'))
@login_required
def thankyou():
  cur_user = g.user
  session.clear()
  logging.info('User %s completed the experiment.' % (cur_user, ))
  return render_template('thankyou.html')
