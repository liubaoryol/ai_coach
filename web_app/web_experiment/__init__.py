'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
import os
from flask import Flask
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
# import eventlet

# eventlet.monkey_patch()

socketio = SocketIO()
db = SQLAlchemy()


def create_app(debug=False, test_config=None):
  """Create an application"""
  app = Flask(__name__, instance_relative_config=True)
  app.debug = debug
  app.config.from_object('config')
  # app.config.from_pyfile('config.py')

  if test_config is None:
    app.config.from_pyfile('config.py', silent=True)
  else:
    app.config.from_mapping(test_config)

  app.config['SQLALCHEMY_DATABASE_URI'] = (
      'sqlite:///' + os.path.abspath(app.config['DATABASE']))

  # ensure the instance folder and data folder exists
  try:
    os.makedirs(app.instance_path)
  except OSError:
    pass

  try:
    os.makedirs(os.path.dirname(app.config['DATABASE']))
  except OSError:
    pass

  db.init_app(app)
  with app.app_context():
    from web_experiment import models  # noqa: F401
    db.create_all()  # Create sql tables for our data models

  from web_experiment.consent import consent_bp
  from web_experiment.auth import auth_bp
  from web_experiment.survey import survey_bp
  from web_experiment.experiment1 import exp1_bp
  from web_experiment.experiment2 import exp2_bp
  app.register_blueprint(consent_bp)
  app.register_blueprint(auth_bp)
  app.register_blueprint(survey_bp)
  app.register_blueprint(exp1_bp)
  app.register_blueprint(exp2_bp)

  # app.add_url_rule('/', endpoint='consent')

  socketio.init_app(app)
  return app
