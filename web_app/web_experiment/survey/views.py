import os
import time
from flask import (flash, g, redirect, render_template, request, url_for,
                   current_app)
from web_experiment.auth.functions import login_required
from web_experiment.models import (db, User, InExperiment, PreExperiment)
from . import survey_bp


@survey_bp.route('/preexperiment', methods=('GET', 'POST'))
@login_required
def preexperiment():
  cur_user = g.user
  if request.method == 'POST':
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
      survey_dir = current_app.config["SURVEY_PATH"]
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
      return redirect(url_for('exp1.tutorial'))

    flash(error)

  query_data = PreExperiment.query.filter_by(subject_id=cur_user).first()
  survey_answers = {
      'age': '',
      'gender': '',
      'never': '',
      'peryear': '',
      'permonth': '',
      'perweek': '',
      'daily': '',
      'precomment': ''
  }
  if query_data is not None:
    survey_answers['age'] = query_data.age
    survey_answers['gender'] = query_data.gender
    survey_answers[query_data.frequency] = 'checked'
    survey_answers['precomment'] = query_data.comment

  return render_template('preexperiment.html', answers=survey_answers)


@survey_bp.route('/survey_both_tell_align', methods=('GET', 'POST'))
@login_required
def survey_both_tell_align():
  EXP_NO = 1
  cur_user = g.user
  if request.method == 'POST':
    maintained = request.form['maintained']
    cooperative = request.form['cooperative']
    fluency = request.form['fluency']
    comments = request.form['opencomment']
    error = None

    if not maintained or not cooperative or not fluency:
      error = 'Please all required questions'

    if error is None:
      # save somewhere
      survey_dir = current_app.config["SURVEY_PATH"]
      if not os.path.exists(survey_dir):
        os.makedirs(survey_dir)

      sec, msec = divmod(time.time() * 1000, 1000)
      time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                              time.gmtime(sec)), msec)
      file_name = (('insurvey%d_' + str(cur_user) + '_' + time_stamp + '.txt') %
                   (EXP_NO, ))
      with open(os.path.join(survey_dir, file_name), 'w',
                newline='') as txtfile:
        txtfile.write('id: ' + cur_user + '\n')
        txtfile.write('maintained: ' + maintained + '\n')
        txtfile.write('cooperative: ' + cooperative + '\n')
        txtfile.write('fluency: ' + fluency + '\n')
        txtfile.write('comments: ' + comments + '\n')
      qdata = InExperiment.query.filter_by(subject_id=cur_user,
                                           exp_number=EXP_NO).first()
      if qdata is not None:
        db.session.delete(qdata)
        db.session.commit()

      new_in_exp = InExperiment(exp_number=EXP_NO,
                                subject_id=cur_user,
                                maintained=maintained,
                                cooperative=cooperative,
                                fluency=fluency,
                                comment=comments)
      db.session.add(new_in_exp)
      db.session.commit()
      return redirect(url_for('exp1.exp1_both_user_random'))

    flash(error)
  query_data = InExperiment.query.filter_by(subject_id=cur_user,
                                            exp_number=1).first()
  survey_answers = {
      'maintainedyes': '',
      'maintainedno': '',
      'cooperativeyes': '',
      'cooperativeno': '',
      'one': '',
      'two': '',
      'three': '',
      'four': '',
      'five': '',
      'incomment': ''
  }
  if query_data is not None:
    survey_answers['maintained' + query_data.maintained] = 'checked'
    survey_answers['cooperative' + query_data.cooperative] = 'checked'
    survey_answers[query_data.fluency] = 'checked'
    survey_answers['incomment'] = query_data.comment

  return render_template('survey_both_tell_align.html', answers=survey_answers)


@survey_bp.route('/survey_both_user_random', methods=('GET', 'POST'))
@login_required
def survey_both_user_random():
  EXP_NO = 2
  cur_user = g.user
  if request.method == 'POST':
    maintained = request.form['maintained']
    cooperative = request.form['cooperative']
    fluency = request.form['fluency']
    comments = request.form['opencomment']
    error = None

    if not maintained or not cooperative or not fluency:
      error = 'Please all required questions'

    if error is None:
      # save somewhere
      survey_dir = current_app.config["SURVEY_PATH"]
      if not os.path.exists(survey_dir):
        os.makedirs(survey_dir)

      sec, msec = divmod(time.time() * 1000, 1000)
      time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                              time.gmtime(sec)), msec)
      file_name = (('insurvey%d_' + str(cur_user) + '_' + time_stamp + '.txt') %
                   (EXP_NO, ))
      with open(os.path.join(survey_dir, file_name), 'w',
                newline='') as txtfile:
        txtfile.write('id: ' + cur_user + '\n')
        txtfile.write('maintained: ' + maintained + '\n')
        txtfile.write('cooperative: ' + cooperative + '\n')
        txtfile.write('fluency: ' + fluency + '\n')
        txtfile.write('comments: ' + comments + '\n')
      qdata = InExperiment.query.filter_by(subject_id=cur_user,
                                           exp_number=EXP_NO).first()
      if qdata is not None:
        db.session.delete(qdata)
        db.session.commit()

      new_in_exp = InExperiment(exp_number=EXP_NO,
                                subject_id=cur_user,
                                maintained=maintained,
                                cooperative=cooperative,
                                fluency=fluency,
                                comment=comments)
      db.session.add(new_in_exp)
      db.session.commit()
      return redirect(url_for('exp1.tutorial2'))

    flash(error)
  query_data = InExperiment.query.filter_by(subject_id=cur_user,
                                            exp_number=EXP_NO).first()
  survey_answers = {
      'maintainedyes': '',
      'maintainedno': '',
      'cooperativeyes': '',
      'cooperativeno': '',
      'one': '',
      'two': '',
      'three': '',
      'four': '',
      'five': '',
      'incomment': ''
  }
  if query_data is not None:
    survey_answers['maintained' + query_data.maintained] = 'checked'
    survey_answers['cooperative' + query_data.cooperative] = 'checked'
    survey_answers[query_data.fluency] = 'checked'
    survey_answers['incomment'] = query_data.comment

  return render_template('survey_both_user_random.html', answers=survey_answers)


@survey_bp.route('/survey_indv_tell_align', methods=('GET', 'POST'))
@login_required
def survey_indv_tell_align():
  EXP_NO = 3
  cur_user = g.user
  if request.method == 'POST':
    maintained = request.form['maintained']
    cooperative = request.form['cooperative']
    fluency = request.form['fluency']
    comments = request.form['opencomment']
    error = None

    if not maintained or not cooperative or not fluency:
      error = 'Please all required questions'

    if error is None:
      # save somewhere
      survey_dir = current_app.config["SURVEY_PATH"]
      if not os.path.exists(survey_dir):
        os.makedirs(survey_dir)

      sec, msec = divmod(time.time() * 1000, 1000)
      time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                              time.gmtime(sec)), msec)
      file_name = (('insurvey%d_' + str(cur_user) + '_' + time_stamp + '.txt') %
                   (EXP_NO, ))
      with open(os.path.join(survey_dir, file_name), 'w',
                newline='') as txtfile:
        txtfile.write('id: ' + cur_user + '\n')
        txtfile.write('maintained: ' + maintained + '\n')
        txtfile.write('cooperative: ' + cooperative + '\n')
        txtfile.write('fluency: ' + fluency + '\n')
        txtfile.write('comments: ' + comments + '\n')
      qdata = InExperiment.query.filter_by(subject_id=cur_user,
                                           exp_number=EXP_NO).first()
      if qdata is not None:
        db.session.delete(qdata)
        db.session.commit()

      new_in_exp = InExperiment(exp_number=EXP_NO,
                                subject_id=cur_user,
                                maintained=maintained,
                                cooperative=cooperative,
                                fluency=fluency,
                                comment=comments)
      db.session.add(new_in_exp)
      db.session.commit()
      return redirect(url_for('exp1.exp1_indv_tell_random'))

    flash(error)
  query_data = InExperiment.query.filter_by(subject_id=cur_user,
                                            exp_number=EXP_NO).first()
  survey_answers = {
      'maintainedyes': '',
      'maintainedno': '',
      'cooperativeyes': '',
      'cooperativeno': '',
      'one': '',
      'two': '',
      'three': '',
      'four': '',
      'five': '',
      'incomment': ''
  }
  if query_data is not None:
    survey_answers['maintained' + query_data.maintained] = 'checked'
    survey_answers['cooperative' + query_data.cooperative] = 'checked'
    survey_answers[query_data.fluency] = 'checked'
    survey_answers['incomment'] = query_data.comment

  return render_template('survey_indv_tell_align.html', answers=survey_answers)


@survey_bp.route('/survey_indv_tell_random', methods=('GET', 'POST'))
@login_required
def survey_indv_tell_random():
  EXP_NO = 4
  cur_user = g.user
  if request.method == 'POST':
    maintained = request.form['maintained']
    cooperative = request.form['cooperative']
    fluency = request.form['fluency']
    comments = request.form['opencomment']
    error = None

    if not maintained or not cooperative or not fluency:
      error = 'Please all required questions'

    if error is None:
      # save somewhere
      survey_dir = current_app.config["SURVEY_PATH"]
      if not os.path.exists(survey_dir):
        os.makedirs(survey_dir)

      sec, msec = divmod(time.time() * 1000, 1000)
      time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                              time.gmtime(sec)), msec)
      file_name = (('insurvey%d_' + str(cur_user) + '_' + time_stamp + '.txt') %
                   (EXP_NO, ))
      with open(os.path.join(survey_dir, file_name), 'w',
                newline='') as txtfile:
        txtfile.write('id: ' + cur_user + '\n')
        txtfile.write('maintained: ' + maintained + '\n')
        txtfile.write('cooperative: ' + cooperative + '\n')
        txtfile.write('fluency: ' + fluency + '\n')
        txtfile.write('comments: ' + comments + '\n')
      qdata = InExperiment.query.filter_by(subject_id=cur_user,
                                           exp_number=EXP_NO).first()
      if qdata is not None:
        db.session.delete(qdata)
        db.session.commit()

      new_in_exp = InExperiment(exp_number=EXP_NO,
                                subject_id=cur_user,
                                maintained=maintained,
                                cooperative=cooperative,
                                fluency=fluency,
                                comment=comments)
      db.session.add(new_in_exp)
      db.session.commit()
      return redirect(url_for('exp1.exp1_indv_user_random'))

    flash(error)
  query_data = InExperiment.query.filter_by(subject_id=cur_user,
                                            exp_number=EXP_NO).first()
  survey_answers = {
      'maintainedyes': '',
      'maintainedno': '',
      'cooperativeyes': '',
      'cooperativeno': '',
      'one': '',
      'two': '',
      'three': '',
      'four': '',
      'five': '',
      'incomment': ''
  }
  if query_data is not None:
    survey_answers['maintained' + query_data.maintained] = 'checked'
    survey_answers['cooperative' + query_data.cooperative] = 'checked'
    survey_answers[query_data.fluency] = 'checked'
    survey_answers['incomment'] = query_data.comment

  return render_template('survey_indv_tell_random.html', answers=survey_answers)


@survey_bp.route('/survey_indv_user_random', methods=('GET', 'POST'))
@login_required
def survey_indv_user_random():
  EXP_NO = 5
  cur_user = g.user
  if request.method == 'POST':
    maintained = request.form['maintained']
    cooperative = request.form['cooperative']
    fluency = request.form['fluency']
    comments = request.form['opencomment']
    error = None

    if not maintained or not cooperative or not fluency:
      error = 'Please all required questions'

    if error is None:
      # save somewhere
      survey_dir = current_app.config["SURVEY_PATH"]
      if not os.path.exists(survey_dir):
        os.makedirs(survey_dir)

      sec, msec = divmod(time.time() * 1000, 1000)
      time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                              time.gmtime(sec)), msec)
      file_name = (('insurvey%d_' + str(cur_user) + '_' + time_stamp + '.txt') %
                   (EXP_NO, ))
      with open(os.path.join(survey_dir, file_name), 'w',
                newline='') as txtfile:
        txtfile.write('id: ' + cur_user + '\n')
        txtfile.write('maintained: ' + maintained + '\n')
        txtfile.write('cooperative: ' + cooperative + '\n')
        txtfile.write('fluency: ' + fluency + '\n')
        txtfile.write('comments: ' + comments + '\n')
      qdata = InExperiment.query.filter_by(subject_id=cur_user,
                                           exp_number=EXP_NO).first()
      if qdata is not None:
        db.session.delete(qdata)
        db.session.commit()

      new_in_exp = InExperiment(exp_number=EXP_NO,
                                subject_id=cur_user,
                                maintained=maintained,
                                cooperative=cooperative,
                                fluency=fluency,
                                comment=comments)
      db.session.add(new_in_exp)
      db.session.commit()
      return redirect(url_for('survey.completion'))

    flash(error)
  query_data = InExperiment.query.filter_by(subject_id=cur_user,
                                            exp_number=EXP_NO).first()
  survey_answers = {
      'maintainedyes': '',
      'maintainedno': '',
      'cooperativeyes': '',
      'cooperativeno': '',
      'one': '',
      'two': '',
      'three': '',
      'four': '',
      'five': '',
      'incomment': ''
  }
  if query_data is not None:
    survey_answers['maintained' + query_data.maintained] = 'checked'
    survey_answers['cooperative' + query_data.cooperative] = 'checked'
    survey_answers[query_data.fluency] = 'checked'
    survey_answers['incomment'] = query_data.comment

  return render_template('survey_indv_user_random.html', answers=survey_answers)


@survey_bp.route('/completion', methods=('GET', 'POST'))
@login_required
def completion():
  cur_user = g.user
  if request.method == 'POST':
    email = request.form['email']
    if email:
      survey_dir = current_app.config["SURVEY_PATH"]
      # save somewhere
      if not os.path.exists(survey_dir):
        os.makedirs(survey_dir)

      sec, msec = divmod(time.time() * 1000, 1000)
      time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                              time.gmtime(sec)), msec)
      file_name = ('email_' + str(cur_user) + '_' + time_stamp + '.txt')
      with open(os.path.join(survey_dir, file_name), 'w',
                newline='') as txtfile:
        txtfile.write('id: ' + cur_user + '\n')
        txtfile.write('email: ' + email + '\n')
      error = None
      user = User.query.filter_by(userid=cur_user).first()

      if user is None:
        error = 'Incorrect user id'

      if error is None:
        user.email = email
        db.session.commit()
        return redirect(url_for('consent.consent'))

      flash(error)
    else:
      return redirect(url_for('consent.consent'))

  query_data = User.query.filter_by(userid=cur_user).first()
  email_db = query_data.email
  return render_template('completion.html', saved_email=email_db)
