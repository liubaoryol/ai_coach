import functools
from flask import g, redirect, session, url_for, request
from . import auth_bp, ADMIN_ID


@auth_bp.before_app_request
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
      return redirect(url_for('consent.consent'))

    return view(**kwargs)

  return wrapped_view


def admin_required(view):
  @functools.wraps(view)
  def wrapped_view(**kwargs):
    if g.user != ADMIN_ID:
      return redirect(url_for('consent.consent'))

    return view(**kwargs)

  return wrapped_view
