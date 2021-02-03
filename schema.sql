DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS demographic;
DROP TABLE IF EXISTS inexperiment;
DROP TABLE IF EXISTS postexperiment;

CREATE TABLE user (
    userid TEXT UNIQUE NOT NULL PRIMARY KEY,
    email TEXT DEFAULT ''
);

CREATE TABLE demographic (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject_id INTEGER NOT NULL,
  age TEXT NOT NULL,
  gender TEXT NOT NULL,
  frequency TEXT NOT NULL,
  precomment TEXT,

  FOREIGN KEY (subject_id) REFERENCES user (userid)
);

CREATE TABLE inexperiment (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject_id INTEGER NOT NULL,
  exp_number INTEGER NOT NULL,
  maintained TEXT NOT NULL,
  cooperative TEXT NOT NULL,
  fluency TEXT NOT NULL,
  incomment TEXT,

  FOREIGN KEY (subject_id) REFERENCES user (userid)
);

CREATE TABLE postexperiment (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject_id INTEGER NOT NULL,
  postcomment TEXT,

  FOREIGN KEY (subject_id) REFERENCES user (userid)
);
