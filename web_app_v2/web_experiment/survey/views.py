from dataclasses import replace
import os
import time
import logging
from flask import (flash, g, redirect, render_template, request, url_for,
                   current_app, session)
from web_experiment.auth.functions import login_required
from web_experiment.models import (db, User, InExperiment, PreExperiment,
                                   PostExperiment)
import csv
from web_experiment.define import (PageKey, get_next_url, ExpType, EDomainType,
                                   get_domain_type)
import web_experiment.exp_intervention.define as intv
import web_experiment.exp_datacollection.define as dcol
from . import survey_bp

SURVEY_TEMPLATE = {
    EDomainType.Movers: 'inexperiment_session_a.html',
    EDomainType.Cleanup: 'inexperiment_session_b.html',
}


def preexperiment():
  cur_user = g.user
  cur_endpoint = survey_bp.name + "." + PageKey.PreExperiment
  group_id = session["groupid"]
  exp_type = session["exp_type"]

  if request.method == 'POST':
    logging.info('User %s submitted pre-experiment survey.' % (cur_user, ))
    age = request.form['age']
    gender = request.form['gender']
    frequency = request.form['frequency']
    comments = request.form['opencomment']
    # getting rid of commas
    # comments = comments.replace(',', ' ')
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

      file_name = ('presurvey_' + str(cur_user) + '_' + time_stamp + '.csv')
      with open(os.path.join(survey_dir, file_name), 'w') as file:
        writer = csv.writer(file)
        header = ['id', 'age', 'gender', 'frequency', 'comments']
        writer.writerow(header)
        row = [cur_user, age, gender, frequency, comments]
        writer.writerow(row)

        # txtfile.write('id: ' + cur_user + '\n')
        # txtfile.write('age: ' + age + '\n')
        # txtfile.write('gender: ' + gender + '\n')
        # txtfile.write('frequency: ' + frequency + '\n')
        # txtfile.write('comments: ' + comments + '\n')

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
      # return redirect(url_for('inst.movers_and_packers'))
      return redirect(get_next_url(cur_endpoint, None, group_id, exp_type))

    flash(error)

  query_data = PreExperiment.query.filter_by(subject_id=cur_user).first()
  survey_answers = {}
  if query_data is not None:
    survey_answers['age'] = query_data.age
    survey_answers['gender'] = query_data.gender
    survey_answers['frequency' + str(query_data.frequency)] = 'checked'
    survey_answers['precomment'] = query_data.comment

  return render_template('preexperiment.html', answers=survey_answers)


def inexp_survey_view(session_name):
  cur_user = g.user
  cur_endpoint = survey_bp.name + "." + PageKey.InExperiment
  group_id = session["groupid"]
  exp_type = session["exp_type"]

  if request.method == 'POST':
    logging.info('User %s submitted the survey for %s.' %
                 (cur_user, session_name))
    maintained = request.form['maintained']
    fluency = request.form['fluency']
    mycarry = request.form['mycarry']
    robotcarry = request.form['robotcarry']
    robotperception = request.form['robotperception']
    cooperative = request.form['cooperative']
    comments = request.form['opencomment']
    # get rid of comma for later entry into csv file
    # comments = comments.replace(",", " ")
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
      # file_name = f'insurvey_{session_name}_{cur_user}_{time_stamp}.txt'
      # with open(os.path.join(survey_dir, file_name), 'w',
      #           newline='') as txtfile:
      #   txtfile.write('id: ' + cur_user + '\n')
      #   txtfile.write('maintained: ' + maintained + '\n')
      #   txtfile.write('fluency: ' + fluency + '\n')
      #   txtfile.write('mycarry: ' + mycarry + '\n')
      #   txtfile.write('robotcarry: ' + robotcarry + '\n')
      #   txtfile.write('robotperception: ' + robotperception + '\n')
      #   txtfile.write('cooperative: ' + cooperative + '\n')
      #   txtfile.write('comments: ' + comments + '\n')
      qdata = InExperiment.query.filter_by(subject_id=cur_user,
                                           session_name=session_name).first()
      if qdata is not None:
        db.session.delete(qdata)
        db.session.commit()
      else:
        file_name = f'insurvey_{cur_user}.csv'
        file_path = os.path.join(survey_dir, file_name)
        file_exist = (os.path.exists(file_path))

        with open(file_path, 'a') as file:
          writer = csv.writer(file)
          # write header if file is just created
          if not file_exist:
            header = [
                'id', 'sessionname', 'timestamp', 'maintained', 'fluency',
                'mycarry', 'robotcarry', 'robotperception', 'cooperative',
                'comments'
            ]
            writer.writerow(header)
          row = [
              cur_user, session_name, time_stamp, maintained, fluency, mycarry,
              robotcarry, robotperception, cooperative, comments
          ]
          writer.writerow(row)

        # txtfile.write('id: ' + cur_user + '\n')
        # txtfile.write('maintained: ' + maintained + '\n')
        # txtfile.write('fluency: ' + fluency + '\n')
        # txtfile.write('mycarry: ' + mycarry + '\n')
        # txtfile.write('robotcarry: ' + robotcarry + '\n')
        # txtfile.write('robotperception: ' + robotperception + '\n')
        # txtfile.write('cooperative: ' + cooperative + '\n')
        # txtfile.write('comments: ' + comments + '\n')

      new_in_exp = InExperiment(session_name=session_name,
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
      # return redirect(url_for(next_endpoint_name, session_name=session_name))
      return redirect(
          get_next_url(cur_endpoint, session_name, group_id, exp_type))

    flash(error)
  query_data = InExperiment.query.filter_by(subject_id=cur_user,
                                            session_name=session_name).first()
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

<<<<<<< HEAD
  return render_template(current_html_file,
                         answers=survey_answers,
                         post_endpoint=url_for(SURVEY_ENDPOINT[session_name]),
                         session_title=td.EXP1_SESSION_TITLE[session_name])


for session_name in SURVEY_PAGENAMES:

  def make_inexp_survey_view(session_name):

    def inexp_survey_view():
      groupid = session["groupid"]
      next_endpoint = get_next_endpoint(session_name, groupid)

      return inexperiment_impl(session_name, SURVEY_TEMPLATE[session_name],
                               next_endpoint)
=======
  if exp_type == ExpType.Data_collection:
    session_title = dcol.SESSION_TITLE[session_name]
  elif exp_type == ExpType.Intervention:
    session_title = intv.SESSION_TITLE[session_name]
>>>>>>> 6f66f7c4c46ec872cb50b9ae26d32019ba31c678

  return render_template(SURVEY_TEMPLATE[get_domain_type(session_name)],
                         answers=survey_answers,
                         session_title=session_title)


def completion():
  cur_user = g.user
  if request.method == 'POST':
    logging.info('User %s submitted post-experiment survey.' % (cur_user, ))
    comment = request.form['comment']
    # comment = comment.replace(',', ' ')
    question = request.form['question']
    email = request.form['email']

    survey_dir = os.path.join(current_app.config["SURVEY_PATH"], cur_user)
    # save somewhere
    if not os.path.exists(survey_dir):
      os.makedirs(survey_dir)

    sec, msec = divmod(time.time() * 1000, 1000)
    time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                            time.gmtime(sec)), msec)
    file_name = ('postsurvey_' + str(cur_user) + '_' + time_stamp + '.csv')
    with open(os.path.join(survey_dir, file_name), 'w') as file:
      writer = csv.writer(file)
      header = ['id', 'email', 'comment', 'question']
      writer.writerow(header)
      row = [cur_user, email, comment, question]
      writer.writerow(row)

    # with open(os.path.join(survey_dir, file_name), 'w', newline='')
    # as txtfile:
    #   txtfile.write('id: ' + cur_user + '\n')
    #   txtfile.write('email: ' + email + '\n')
    #   txtfile.write('comment: ' + comment + '\n')
    #   txtfile.write('question: ' + question + '\n')
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


def thankyou():
  cur_user = g.user
  session.clear()
  logging.info('User %s completed the experiment.' % (cur_user, ))
  return render_template('thankyou.html')


survey_bp.add_url_rule('/' + PageKey.PreExperiment,
                       PageKey.PreExperiment,
                       login_required(preexperiment),
                       methods=('GET', 'POST'))

survey_bp.add_url_rule('/' + PageKey.InExperiment + "/<session_name>",
                       PageKey.InExperiment,
                       login_required(inexp_survey_view),
                       methods=('GET', 'POST'))

survey_bp.add_url_rule('/' + PageKey.Completion,
                       PageKey.Completion,
                       login_required(completion),
                       methods=('GET', 'POST'))

survey_bp.add_url_rule('/' + PageKey.Thankyou,
                       PageKey.Thankyou,
                       login_required(thankyou),
                       methods=('GET', 'POST'))
