from web_experiment import db


class User(db.Model):
  userid = db.Column(db.String(80),
                     unique=True,
                     nullable=False,
                     primary_key=True)
  email = db.Column(db.String(120), default='')
  admin = db.Column(db.Boolean, nullable=False, default=False)
  pre_exp = db.relationship('PreExperiment',
                            backref='user',
                            lazy=True,
                            uselist=False)
  in_exp = db.relationship('InExperiment',
                           backref='user',
                           lazy=True,
                           uselist=False)
  post_exp = db.relationship('PostExperiment',
                             backref='user',
                             lazy=True,
                             uselist=False)

  def __repr__(self):
    return '<User %r>' % self.userid


class PreExperiment(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  age = db.Column(db.String(80), nullable=False)
  gender = db.Column(db.String(80), nullable=False)
  frequency = db.Column(db.String(80), nullable=False)
  comment = db.Column(db.String(500))
  subject_id = db.Column(db.String(80),
                         db.ForeignKey('user.userid'),
                         nullable=False)

  def __repr__(self):
    return '<PreExperiment %r>' % self.subject_id


class InExperiment(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  exp_number = db.Column(db.Integer, nullable=False)
  maintained = db.Column(db.String(80), nullable=False)
  cooperative = db.Column(db.String(80), nullable=False)
  fluency = db.Column(db.String(80), nullable=False)
  comment = db.Column(db.String(500))
  subject_id = db.Column(db.String(80),
                         db.ForeignKey('user.userid'),
                         nullable=False)

  def __repr__(self):
    return '<InExperiment %r>' % self.subject_id


class PostExperiment(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  comment = db.Column(db.String(500))
  subject_id = db.Column(db.String(80),
                         db.ForeignKey('user.userid'),
                         nullable=False)

  def __repr__(self):
    return '<PostExperiment %r>' % self.subject_id
