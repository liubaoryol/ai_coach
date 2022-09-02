'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint
from web_experiment.define import BPName

review_bp = Blueprint(BPName.Review,
                      __name__,
                      template_folder='templates',
                      static_folder='static',
                      static_url_path='/review/static')

from . import views  # noqa: E402, F401, E501
from . import event_replay, event_record, event_review  # noqa: E402, F401, E501
