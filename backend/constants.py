import json


with open('config.json') as f:
    config = json.load(f)

TRAJECTORY_DIR = config["trajectorypath"]
DATABASE_DIR = config["dbpath"]
SURVEY_DIR = config["surveypath"]