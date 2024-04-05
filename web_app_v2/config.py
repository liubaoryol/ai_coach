SQLALCHEMY_TRACK_MODIFICATIONS = False

# copy config.py to the instance directory and redefine below variables
SECRET_KEY = 'TIC_human_study'

DATA_DIR = "./data"
DATABASE = DATA_DIR + "/tw2020_db/tw2020.sqlite"
SURVEY_PATH = DATA_DIR + "/tw2020_survey"
TRAJECTORY_PATH = DATA_DIR + "/tw2020_trajectory"
LATENT_PATH = DATA_DIR + "/tw2020_latent"
USER_LABEL_PATH = DATA_DIR + "/tw2020_user_label"
EXP_TYPE = "intervention"  # data_collection | intervention

USE_IDENTIFIABLE_URL = True
REMOVE_HISTORY = False
COMPLETION_CODE = 'C1CY2NS8'
COMPLETION_REDIRECT = ("https://app.prolific.com/submissions/complete?cc=" +
                       COMPLETION_CODE)

DEBUG = False
