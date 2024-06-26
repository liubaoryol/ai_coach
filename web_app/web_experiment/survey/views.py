import os
import time
import logging
from flask import (flash, g, redirect, render_template, request, url_for,
                   current_app, session)
from web_experiment.auth.functions import login_required
from web_experiment.models import (db, User, InExperiment, PreExperiment,
                                   PostExperiment)
import web_experiment.experiment1.task_define as td
import web_experiment.survey.names as sn
from . import survey_bp


@survey_bp.route('/preexperiment', methods=('GET', 'POST'))
@login_required
def preexperiment():
  cur_user = g.user
  if request.method == 'POST':
    logging.info('User %s submitted pre-experiment survey.' % (cur_user, ))
    age = request.form['age']
    gender = request.form['gender']
    frequency = request.form['frequency']
    comments = request.form['opencomment']
    error = None

    if not age:
      error = 'Age is required'
    elif not gender:
      error = 'Gender is required'
    elif not frequency:
      error = 'Please select how often you play video games'

    if error is None:
      # save somewhere
      survey_dir = os.path.join(current_app.config["SURVEY_PATH"], cur_user)
      if not os.path.exists(survey_dir):
        os.makedirs(survey_dir)

      sec, msec = divmod(time.time() * 1000, 1000)
      time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                              time.gmtime(sec)), msec)
      file_name = ('presurvey1_' + str(cur_user) + '_' + time_stamp + '.txt')
      with open(os.path.join(survey_dir, file_name), 'w',
                newline='') as txtfile:
        txtfile.write('id: ' + cur_user + '\n')
        txtfile.write('age: ' + age + '\n')
        txtfile.write('gender: ' + gender + '\n')
        txtfile.write('frequency: ' + frequency + '\n')
        txtfile.write('comments: ' + comments + '\n')

      qdata = PreExperiment.query.filter_by(subject_id=cur_user).first()
      if qdata is not None:
        db.session.delete(qdata)
        db.session.commit()

      new_pre_exp = PreExperiment(age=age,
                                  subject_id=cur_user,
                                  gender=gender,
                                  frequency=frequency,
                                  comment=comments)
      db.session.add(new_pre_exp)
      db.session.commit()
      return redirect(url_for('inst.movers_and_packers'))

    flash(error)

  query_data = PreExperiment.query.filter_by(subject_id=cur_user).first()
  survey_answers = {}
  if query_data is not None:
    survey_answers['age'] = query_data.age
    survey_answers['gender'] = query_data.gender
    survey_answers['frequency' + str(query_data.frequency)] = 'checked'
    survey_answers['precomment'] = query_data.comment

  return render_template('preexperiment.html', answers=survey_answers)


def inexperiment_impl(exp_no, current_html_file, next_endpoint_name):
  cur_user = g.user
  if request.method == 'POST':
    logging.info('User %s submitted in-experiment survey #%d.' %
                 (cur_user, exp_no))
    maintained = request.form['maintained']
    fluency = request.form['fluency']
    mycarry = request.form['mycarry']
    robotcarry = request.form['robotcarry']
    robotperception = request.form['robotperception']
    cooperative = request.form['cooperative']
    comments = request.form['opencomment']
    error = None

    if (not maintained or not cooperative or not fluency or not mycarry
        or not robotcarry or not robotperception):
      error = 'Please answer all required questions'

    if error is None:
      # save somewhere
      survey_dir = os.path.join(current_app.config["SURVEY_PATH"], cur_user)
      if not os.path.exists(survey_dir):
        os.makedirs(survey_dir)

      sec, msec = divmod(time.time() * 1000, 1000)
      time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                              time.gmtime(sec)), msec)
      file_name = (('insurvey%d_' + str(cur_user) + '_' + time_stamp + '.txt') %
                   (exp_no, ))
      with open(os.path.join(survey_dir, file_name), 'w',
                newline='') as txtfile:
        txtfile.write('id: ' + cur_user + '\n')
        txtfile.write('maintained: ' + maintained + '\n')
        txtfile.write('fluency: ' + fluency + '\n')
        txtfile.write('mycarry: ' + mycarry + '\n')
        txtfile.write('robotcarry: ' + robotcarry + '\n')
        txtfile.write('robotperception: ' + robotperception + '\n')
        txtfile.write('cooperative: ' + cooperative + '\n')
        txtfile.write('comments: ' + comments + '\n')
      qdata = InExperiment.query.filter_by(subject_id=cur_user,
                                           exp_number=exp_no).first()
      if qdata is not None:
        db.session.delete(qdata)
        db.session.commit()

      new_in_exp = InExperiment(exp_number=exp_no,
                                subject_id=cur_user,
                                maintained=maintained,
                                mycarry=mycarry,
                                robotcarry=robotcarry,
                                robotperception=robotperception,
                                cooperative=cooperative,
                                fluency=fluency,
                                comment=comments)
      db.session.add(new_in_exp)
      db.session.commit()
      return redirect(url_for(next_endpoint_name))

    flash(error)
  query_data = InExperiment.query.filter_by(subject_id=cur_user,
                                            exp_number=exp_no).first()
  survey_answers = {}
  if query_data is not None:
    survey_answers['maintained' + str(query_data.maintained)] = 'checked'
    survey_answers['cooperative' + str(query_data.cooperative)] = 'checked'
    survey_answers['fluency' + str(query_data.fluency)] = 'checked'
    survey_answers['mycarry' + str(query_data.mycarry)] = 'checked'
    survey_answers['robotcarry' + str(query_data.robotcarry)] = 'checked'
    survey_answers['robotperception' +
                   str(query_data.robotperception)] = 'checked'
    survey_answers['incomment'] = query_data.comment

  session_name = td.LIST_SESSIONS[exp_no - 1]
  return render_template(current_html_file,
                         answers=survey_answers,
                         post_endpoint=url_for(
                             sn.SURVEY_ENDPOINT[session_name]),
                         session_title=td.EXP1_SESSION_TITLE[session_name])


for idx in range(len(td.LIST_SESSIONS)):

  session_name = td.LIST_SESSIONS[idx]

  def make_inexp_survey_view(idx, session_name=session_name):
    def inexp_survey_view():
      return inexperiment_impl(idx + 1, sn.SURVEY_TEMPLATE[session_name],
                               sn.SURVEY_NEXT_ENDPOINT[session_name])

    return inexp_survey_view

  func = login_required(make_inexp_survey_view(idx))
  survey_bp.add_url_rule('/' + sn.SURVEY_PAGENAMES[session_name],
                         sn.SURVEY_PAGENAMES[session_name],
                         func,
                         methods=('GET', 'POST'))


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
