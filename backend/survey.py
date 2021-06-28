import os
import time
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from backend.constants import SURVEY_DIR
from backend.db import get_db, query_db
from backend.auth import login_required


bp = Blueprint('survey', __name__)


@bp.route('/preexperiment', methods=('GET', 'POST'))
@login_required
def preexperiment():
    db = get_db()
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
            if not os.path.exists(SURVEY_DIR):
                os.makedirs(SURVEY_DIR)

            sec, msec = divmod(time.time() * 1000, 1000)
            time_stamp = '%s.%03d' % (
                time.strftime('%Y-%m-%d_%H_%M_%S', time.gmtime(sec)), msec)
            file_name = ('presurvey1_' + str(cur_user) + '_' + time_stamp + '.txt')
            with open(os.path.join(SURVEY_DIR, file_name),
                    'w', newline='') as txtfile:
                txtfile.write('id: ' + cur_user + '\n')
                txtfile.write('age: ' + age + '\n')
                txtfile.write('gender: ' + gender + '\n')
                txtfile.write('frequency: ' + frequency + '\n')
                txtfile.write('comments: ' + comments + '\n')

            qdata = query_db(
                'SELECT * FROM demographic WHERE subject_id = ?',
                (cur_user,), one=True)
            if qdata is not None:
                db.execute('DELETE FROM demographic WHERE subject_id = ?', (cur_user,))
                db.commit()

            db.execute(
                'INSERT INTO demographic'
                ' (subject_id, age, gender, frequency, precomment)'
                ' VALUES (?, ?, ?, ?, ?)',
                (cur_user, age, gender, frequency, comments))
            db.commit()
            return redirect(url_for('instruction'))

        flash(error)

    query_data = query_db(
            'SELECT * FROM demographic WHERE subject_id = ?',
            (cur_user,), one=True
        )
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
        survey_answers['age'] = query_data['age']
        survey_answers['gender'] = query_data['gender']
        survey_answers[query_data['frequency']] = 'checked'
        survey_answers['precomment'] = query_data['precomment']

    return render_template('preexperiment.html', answers=survey_answers)


@bp.route('/inexperiment', methods=('GET', 'POST'))
@login_required
def inexperiment():
    db = get_db()
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
            if not os.path.exists(SURVEY_DIR):
                os.makedirs(SURVEY_DIR)

            sec, msec = divmod(time.time() * 1000, 1000)
            time_stamp = '%s.%03d' % (
                time.strftime('%Y-%m-%d_%H_%M_%S', time.gmtime(sec)), msec)
            file_name = ('insurvey1_'  + str(cur_user) + '_' + time_stamp + '.txt')
            with open(os.path.join(SURVEY_DIR, file_name),
                    'w', newline='') as txtfile:
                txtfile.write('id: ' + cur_user + '\n')
                txtfile.write('maintained: ' + maintained + '\n')
                txtfile.write('cooperative: ' + cooperative + '\n')
                txtfile.write('fluency: ' + fluency + '\n')
                txtfile.write('comments: ' + comments + '\n')

            qdata = query_db(
                'SELECT * FROM inexperiment WHERE subject_id = ? AND exp_number = ?',
                (cur_user, 1), one=True)
            if qdata is not None:
                db.execute(
                    'DELETE FROM inexperiment WHERE subject_id = ? AND exp_number = ?',
                    (cur_user, 1))
                db.commit()

            db.execute(
                'INSERT INTO inexperiment'
                ' (subject_id, exp_number, maintained, cooperative, fluency, incomment)'
                ' VALUES (?, 1, ?, ?, ?, ?)',
                (cur_user, maintained, cooperative, fluency, comments))
            db.commit()
            return redirect(url_for('survey.completion'))

        flash(error)

    query_data = query_db(
            'SELECT * FROM inexperiment WHERE subject_id = ? AND exp_number = ?',
            (cur_user, 1), one=True
        )
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
        survey_answers['maintained' + query_data['maintained']] = 'checked'
        survey_answers['cooperative' + query_data['cooperative']] = 'checked'
        survey_answers[query_data['fluency']] = 'checked'
        survey_answers['incomment'] = query_data['incomment']

    return render_template('inexperiment.html', answers=survey_answers)


@bp.route('/completion', methods=('GET', 'POST'))
@login_required
def completion():
    db = get_db()
    cur_user = g.user
    if request.method == 'POST':
        email = request.form['email']
        if email:
            # save somewhere
            if not os.path.exists(SURVEY_DIR):
                os.makedirs(SURVEY_DIR)

            sec, msec = divmod(time.time() * 1000, 1000)
            time_stamp = '%s.%03d' % (
                time.strftime('%Y-%m-%d_%H_%M_%S', time.gmtime(sec)), msec)
            file_name = ('email_'  + str(cur_user) + '_' + time_stamp + '.txt')
            with open(os.path.join(SURVEY_DIR, file_name),
                    'w', newline='') as txtfile:
                txtfile.write('id: ' + cur_user + '\n')
                txtfile.write('email: ' + email + '\n')
            error = None
            user = query_db(
                'SELECT * FROM user WHERE userid = ?', (cur_user,), one=True)

            if user is None:
                error = 'Incorrect user id'

            if error is None:
                db.execute(
                    'UPDATE user SET email = ? WHERE userid = ?',
                    (email, cur_user))
                db.commit()
                return redirect(url_for('consent'))

            flash(error)
        else:
            return redirect(url_for('consent'))

    query_data = query_db(
            'SELECT * FROM user WHERE userid = ?',
            (cur_user,), one=True
        )
    email_db = query_data['email']
    return render_template('completion.html', saved_email=email_db)
