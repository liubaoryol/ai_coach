import functools
import os
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from db import get_db, query_db
bp = Blueprint('auth', __name__)

ADMIN_ID = 'register1234'

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = user_id

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('consent'))

        return view(**kwargs)

    return wrapped_view

def admin_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user != ADMIN_ID:
            return redirect(url_for('consent'))

        return view(**kwargs)

    return wrapped_view

def write_id(uid):
    file_name = "users.txt"

    with open(file_name, 'a', newline='') as txtfile:
        txtfile.write(uid)
        txtfile.write("\n")

def query_id():
    file_name = "users.txt"
    ids = ['ddd']
    if not os.path.exists(file_name):
        return ids

    with open(file_name, newline='') as txtfile:
        rows = txtfile.readlines()
        for row in rows:
            ids.append(row.rstrip())
    return ids

@bp.route('/', methods=('GET', 'POST'))
def consent():
    session.clear()
    if request.method == 'POST':
        userid = request.form['userid']
        if userid == ADMIN_ID:
            session['user_id'] = userid
            return redirect(url_for("auth.register"))

        error = None
        db = get_db()
        user = query_db(
            'SELECT * FROM user WHERE userid = ?', (userid,), one=True)

        if user is None:
            error = 'Incorrect user id'
        
        if error is None:
            session['user_id'] = user['userid']
            return redirect(url_for("survey.preexperiment"))

        flash(error)
 
    return render_template('consent.html')

@bp.route('/register', methods=('GET', 'POST'))
@admin_required
def register():
    db = get_db()
    if request.method == 'POST':
        if 'userid' in request.form:
            userid = request.form['userid']
            error = None

            if not userid:
                error = 'User ID is required.'
            else:
                qdata = query_db(
                    'SELECT * FROM user WHERE userid = ?', (userid,), one=True
                    )
                if qdata is not None:
                    error = 'User {} is already registered.'.format(userid)

            if error is None:
                db.execute('INSERT INTO user (userid) VALUES (?)', (userid,))
                db.commit()
                return redirect(url_for('auth.register'))

            flash(error)
        elif 'delid' in request.form:
            delid = request.form['delid']
            db.execute('DELETE FROM user WHERE userid = ?', (delid,))
            db.commit()
            return redirect(url_for('auth.register'))
    
    user_list = query_db( 'SELECT * FROM user')

    return render_template('register.html', userlist=user_list)
