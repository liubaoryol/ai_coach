'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
import os
import logging
from flask import Flask
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
# import eventlet

# eventlet.monkey_patch()

socketio = SocketIO()
db = SQLAlchemy()


def create_app(debug=False, test_config=None):
  logging.basicConfig(
      filename='myapp.log',
      level=logging.INFO,
      format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
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
  app.config.update(SESSION_COOKIE_SAMESITE='None',
                    SESSION_COOKIE_SECURE='True')

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
  from web_experiment.consent.views import consent
  from web_experiment.auth import auth_bp
  from web_experiment.survey import survey_bp
  from web_experiment.exp_intervention import exp_interv_bp
  from web_experiment.exp_datacollection import exp_dcollect_bp
  from web_experiment.instruction import inst_bp
  from web_experiment.review import review_bp
  from web_experiment.feedback import feedback_bp
  from web_experiment.demo import demo_bp
  app.register_blueprint(consent_bp)
  app.register_blueprint(auth_bp)
  app.register_blueprint(survey_bp)
  app.register_blueprint(exp_interv_bp)
  app.register_blueprint(exp_dcollect_bp)
  app.register_blueprint(inst_bp)
  app.register_blueprint(review_bp)
  app.register_blueprint(feedback_bp)
  app.register_blueprint(demo_bp)

  app.add_url_rule('/', 'index', consent, methods=("GET", "POST"))
  # app.add_url_rule('/', 'index', index, methods=("GET", "POST"))

  socketio.init_app(app)

  logging.info('Create app!')
  return app
