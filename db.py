import sqlite3
import os

import click
from flask import current_app, g
from flask.cli import with_appcontext

# DATABASE_DIR = '/data/tw2020_db'
DATABASE_DIR = './data/tw2020_db'
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

# def get_db():
#     if 'db' not in g:
#         g.db = sqlite3.connect(
#             current_app.config['DATABASE'],
#             detect_types=sqlite3.PARSE_DECLTYPES
#         )
#         g.db.row_factory = sqlite3.Row
#     return g.db

# def close_db(e=None):
#     db = g.pop('db', None)
#     if db is not None:
#         db.close()

# @click.command('init-db')
# @with_appcontext
# def init_db_command():
#     """Clear the existing data and create new tables."""
#     init_db()
#     click.echo('Initialized the database.')

def init_app(app):
    app.teardown_appcontext(close_connection)
    # app.cli.add_command(init_db_command)
