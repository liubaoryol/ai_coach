'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint

survey_bp = Blueprint('survey',
                      __name__,
                      template_folder='templates',
                      static_folder='static',
                      static_url_path='/survey/static')

from . import views  # noqa: E402, F401
