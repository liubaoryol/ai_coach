import os
import time
import logging
from flask import (flash, g, redirect, render_template, request, url_for,
                   current_app, session)
from web_experiment.auth.functions import login_required
from web_experiment.models import (db, User, InExperiment, PreExperiment,
                                   PostExperiment)
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


def inexperiment_impl(exp_no, current_html_file, next_endpoint_name, session_name = None):
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
      if (session_name):
        return redirect(url_for(next_endpoint_name, session_name = session_name))

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

  return render_template(current_html_file, answers=survey_answers)


@survey_bp.route('/survey_both_tell_align', methods=('GET', 'POST'))
@login_required
def survey_both_tell_align():
  return inexperiment_impl(1, 'survey_both_tell_align.html',
                           'exp1.exp1_trial1')


@survey_bp.route('/survey_trial1', methods=('GET', 'POST'))
@login_required
def survey_trial1():
  if session['user_group'] == 'C':
    return inexperiment_impl(3, 'survey_trial1.html',
                           'feedback.collect', session_name = "a3")
  else:
    return inexperiment_impl(3, 'survey_trial1.html',
                           'exp1.exp1_both_user_random_2')


@survey_bp.route('/survey_both_user_random_2', methods=('GET', 'POST'))
@login_required
def survey_both_user_random_2():
  return inexperiment_impl(4, 'survey_both_user_random_2.html', 'inst.clean_up')


@survey_bp.route('/survey_indv_tell_align', methods=('GET', 'POST'))
@login_required
def survey_indv_tell_align():
  return inexperiment_impl(5, 'survey_indv_tell_align.html',
                           'exp1.exp1_indv_tell_random')


@survey_bp.route('/survey_indv_tell_random', methods=('GET', 'POST'))
@login_required
def survey_indv_tell_random():
  return inexperiment_impl(6, 'survey_indv_tell_random.html',
                           'exp1.exp1_indv_user_random')


@survey_bp.route('/survey_indv_user_random', methods=('GET', 'POST'))
@login_required
def survey_indv_user_random():
  return inexperiment_impl(7, 'survey_indv_user_random.html',
                           'exp1.exp1_indv_user_random_2')


@survey_bp.route('/survey_indv_user_random_2', methods=('GET', 'POST'))
@login_required
def survey_indv_user_random_2():
  return inexperiment_impl(8, 'survey_indv_user_random_2.html',
                           'exp1.exp1_indv_user_random_3')


@survey_bp.route('/survey_indv_user_random_3', methods=('GET', 'POST'))
@login_required
def survey_indv_user_random_3():
  return inexperiment_impl(9, 'survey_indv_user_random_3.html',
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
