import logging
from flask import (flash, redirect, render_template, request, session, url_for,
                   current_app)
from web_experiment.models import User, db
from web_experiment.define import ExpType, PageKey, get_next_url
from web_experiment.auth import ADMIN_ID
from . import consent_bp


def consent():
  session.clear()
  cur_endpoint = consent_bp.name + "." + PageKey.Consent
  account_id = ''
  if request.method == 'POST':
    userid = request.form['userid'].lower()
    account_id = request.form['account']
    if current_app.config['EXP_TYPE'] == 'data_collection':
      session['exp_type'] = ExpType.Data_collection
    elif current_app.config['EXP_TYPE'] == 'intervention':
      session['exp_type'] = ExpType.Intervention

    logging.info('User %s attempts to log in' % (userid, ))
    if userid == ADMIN_ID:
      session['user_id'] = userid
      return redirect(url_for("auth.register"))

    error = None
    user = User.query.filter_by(userid=userid).first()

    if user is None:
      error = 'Unregistered Participation Code'

    if error is None:
      session['user_id'] = user.userid
      session['groupid'] = user.groupid
      if user.completed:
        error = "You already completed the experiment."
        return redirect(url_for('survey.thankyou'))
      else:
        user.account_id = account_id
        db.session.commit()
        return redirect(
            get_next_url(cur_endpoint, None, user.groupid, session['exp_type']))

    flash(error)

  # ?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
  if account_id == '':
    account_id = request.args.get('PROLIFIC_PID')
    if account_id is None:
      account_id = ''

  return render_template('consent.html',
                         cur_endpoint=cur_endpoint,
                         account_id=account_id)


consent_bp.add_url_rule('/' + PageKey.Consent,
                        PageKey.Consent,
                        consent,
                        methods=('GET', 'POST'))
