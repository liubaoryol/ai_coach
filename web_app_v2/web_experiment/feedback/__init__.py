'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint

feedback_bp = Blueprint('feedback',
                        __name__,
                        template_folder='templates',
                        static_folder='static',
                        static_url_path='/feedback/static')

from . import views, helper  # noqa: E402, F401, E501
from . import event_feedback, event_collect_latent  # noqa: E402, F401
