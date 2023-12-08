SQLALCHEMY_TRACK_MODIFICATIONS = False

# copy config.py to the instance directory and redefine below variables
SECRET_KEY = 'sseo_data_col_v2'

DATA_DIR = "./data"
DATABASE = DATA_DIR + "/tw2020_db/tw2020.sqlite"
SURVEY_PATH = DATA_DIR + "/tw2020_survey"
TRAJECTORY_PATH = DATA_DIR + "/tw2020_trajectory"
LATENT_PATH = DATA_DIR + "/tw2020_latent"
USER_LABEL_PATH = DATA_DIR + "/tw2020_user_label"
EXP_TYPE = "data_collection"  # data_collection | intervention

DEBUG = False
