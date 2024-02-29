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
  account_input = ''
  if request.method == 'POST':
    userid = request.form['userid'].lower()
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

    account_input = request.form['account']
    if user is None:
      error = 'Unregistered Participation Code'
    # user is not yet associated with prolific id and
    elif user.account_id == '':
      # the accessed link doesn't contain prolific info.
      if account_input == '':
        # error
        error = ("This session is missing Prolific info. "
                 "Please access again using the link provided by Prolific.")
      # associate prolific id with the user
      else:
        user.account_id = account_input
        db.session.commit()

    if error is None:
      session['user_id'] = user.userid
      session['groupid'] = user.groupid
      if user.completed:
        error = "You've already completed the experiment."
        return redirect(url_for('survey.thankyou'))
      else:
        return redirect(
            get_next_url(cur_endpoint, None, user.groupid, session['exp_type']))

    flash(error)

  # ?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
  if account_input == '':
    account_input = request.args.get('PROLIFIC_PID')
    if account_input is None:
      account_input = ''

  return render_template('consent.html',
                         cur_endpoint=cur_endpoint,
                         prolific_info=account_input)


consent_bp.add_url_rule('/' + PageKey.Consent,
                        PageKey.Consent,
                        consent,
                        methods=('GET', 'POST'))
