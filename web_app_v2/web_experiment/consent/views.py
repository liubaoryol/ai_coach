import logging
from flask import flash, redirect, render_template, request, session, url_for
from web_experiment.models import User
from web_experiment.define import ExpType, PageKey, get_next_url
from web_experiment.auth import ADMIN_ID
from . import consent_bp


def consent():
  session.clear()
  if request.method == 'POST':
    userid = request.form['userid'].lower()
    if request.form['exp_type'] == 'data_collection':
      session['exp_type'] = ExpType.Data_collection
    elif request.form['exp_type'] == 'intervention':
      session['exp_type'] = ExpType.Intervention

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
        session['groupid'] = user.groupid
        cur_endpoint = consent_bp.name + "." + PageKey.Consent
        return redirect(
            get_next_url(cur_endpoint, None, user.groupid, session['exp_type']))

    flash(error)

  return render_template('consent.html')


consent_bp.add_url_rule('/' + PageKey.Consent,
                        PageKey.Consent,
                        consent,
                        methods=('GET', 'POST'))
