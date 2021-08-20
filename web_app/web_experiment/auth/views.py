from flask import (flash, redirect, render_template, request, url_for)
from web_experiment.models import db, User
from web_experiment.auth.functions import admin_required
from . import auth_bp


@auth_bp.route('/register', methods=('GET', 'POST'))
@admin_required
def register():
  if request.method == 'POST':
    if 'userid' in request.form:
      userid = request.form['userid']
      error = None

      if not userid:
        error = 'User ID is required.'
      else:
        qdata = User.query.filter_by(userid=userid).first()
        if qdata is not None:
          error = 'User {} is already registered.'.format(userid)

      if error is None:
        new_user = User(userid=userid)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('auth.register'))

      flash(error)
    elif 'delid' in request.form:
      delid = request.form['delid']
      qdata = User.query.filter_by(userid=delid).first()
      db.session.delete(qdata)
      db.session.commit()

      return redirect(url_for('auth.register'))

  user_list = User.query.all()

  return render_template('register.html', userlist=user_list)
