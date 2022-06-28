import logging
from flask import flash, redirect, render_template, request, session, url_for
# from web_experiment.db import query_db
from web_experiment.models import User
from web_experiment.auth import ADMIN_ID
from . import consent_bp


@consent_bp.route('/', methods=('GET', 'POST'))
def consent():
  session.clear()
  if request.method == 'POST':
    userid = request.form['userid'].lower()
    logging.info('User %s attempts to log in' % (userid, ))
    if userid == ADMIN_ID:
      session['user_id'] = userid
      return redirect(url_for("auth.register"))

    error = None
    user = User.query.filter_by(userid=userid).first()

    if user is None:
      error = 'Incorrect user id'

    if error is None:
      if user.completed:
        error = "You already completed the experiment."
      else:
        session['user_id'] = user.userid
        session['user_group'] = request.form['usergroup']
        print(session['user_group'])
        return redirect(url_for("inst.overview"))

    flash(error)

  return render_template('consent.html')
