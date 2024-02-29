from web_experiment import db
from web_experiment.define import (DATACOL_TUTORIALS, DATACOL_TASKS,
                                   INTERV_SESSIONS, PageKey,
                                   get_record_session_key)
from web_experiment.survey_def import POST_TASK_QUESTIONS


class User(db.Model):
  userid = db.Column(db.String(80),
                     unique=True,
                     nullable=False,
                     primary_key=True)
  groupid = db.Column(db.String(80), unique=False, default='')
  account_id = db.Column(db.String(80), unique=False, default='')
  email = db.Column(db.String(120), default='')
  test = db.Column(db.Boolean, nullable=False, default=False)
  completed = db.Column(db.Boolean, default=False)

  best_a = db.Column(db.Integer, default=999)
  best_b = db.Column(db.Integer, default=999)
  best_c = db.Column(db.Integer, default=0)

  exp_datacollection = db.relationship('ExpDataCollection',
                                       backref='user',
                                       lazy=True,
                                       uselist=False,
                                       passive_deletes=True)
  exp_intervention = db.relationship('ExpIntervention',
                                     backref='user',
                                     lazy=True,
                                     uselist=False,
                                     passive_deletes=True)
  pre_exp = db.relationship('PreExperiment',
                            backref='user',
                            lazy=True,
                            uselist=False,
                            passive_deletes=True)
  in_exp = db.relationship('InExperiment',
                           backref='user',
                           lazy=True,
                           passive_deletes=True)
  post_exp = db.relationship('PostExperiment',
                             backref='user',
                             lazy=True,
                             uselist=False,
                             passive_deletes=True)

  def __repr__(self):
    return '<User %r>' % self.userid


class ExpDataCollection(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  subject_id = db.Column(db.String(80),
                         db.ForeignKey('user.userid', ondelete='CASCADE'),
                         nullable=False)

  for session_name in DATACOL_TUTORIALS:
    vars()[session_name] = db.Column(db.Boolean, default=False)

  for session_name in DATACOL_TASKS:
    vars()[session_name] = db.Column(db.Boolean, default=False)
    vars()[get_record_session_key(session_name)] = db.Column(db.Boolean,
                                                             default=False)

  def __repr__(self):
    return '<ExpDataCollection %r>' % self.subject_id


class ExpIntervention(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  subject_id = db.Column(db.String(80),
                         db.ForeignKey('user.userid', ondelete='CASCADE'),
                         nullable=False)

  for session_name in INTERV_SESSIONS:
    vars()[session_name] = db.Column(db.Boolean, default=False)

  def __repr__(self):
    return '<ExpIntervention %r>' % self.subject_id


class PreExperiment(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  age = db.Column(db.String(80), nullable=False)
  gender = db.Column(db.String(80), nullable=False)
  frequency = db.Column(db.Integer, nullable=False)
  comment = db.Column(db.String(500))
  subject_id = db.Column(db.String(80),
                         db.ForeignKey('user.userid', ondelete='CASCADE'),
                         nullable=False)

  def __repr__(self):
    return '<PreExperiment %r>' % self.subject_id


class InExperiment(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  session_name = db.Column(db.String(80), default='')
  subject_id = db.Column(db.String(80),
                         db.ForeignKey('user.userid', ondelete='CASCADE'),
                         nullable=False)

  for e_question in POST_TASK_QUESTIONS:
    vars()[e_question.name] = db.Column(db.Integer, nullable=True)

  comment = db.Column(db.String(500))

  def __repr__(self):
    return '<InExperiment %r>' % self.subject_id


class PostExperiment(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  comment = db.Column(db.String(500))
  question = db.Column(db.String(500))
  subject_id = db.Column(db.String(80),
                         db.ForeignKey('user.userid', ondelete='CASCADE'),
                         nullable=False)

  def __repr__(self):
    return '<PostExperiment %r>' % self.subject_id
