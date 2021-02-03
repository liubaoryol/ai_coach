import sqlite3
import os
from flask import current_app, g
from flask.cli import with_appcontext


DATABASE_DIR = '/data/tw2020_db'
DATABASE_FILE = os.path.join(DATABASE_DIR, 'teamwork2020.db')


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE_FILE)
        db.row_factory = sqlite3.Row
    return db


def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv


def init_app(app):
    app.teardown_appcontext(close_connection)
    # app.cli.add_command(init_db_command)
