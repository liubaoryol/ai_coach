SQLALCHEMY_TRACK_MODIFICATIONS = False

# copy config.py to the instance directory and redefine below variables
SECRET_KEY = 'sseo_data_col_v2'

DATABASE = "./data/tw2020_db/tw2020.sqlite"
SURVEY_PATH = "./data/tw2020_survey"
TRAJECTORY_PATH = "./data/tw2020_trajectory"
LATENT_PATH = "./data/tw2020_latent"
EXP_TYPE = "intervention"  # data_collection | intervention

DEBUG = False
