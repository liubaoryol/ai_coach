'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint

auth_bp = Blueprint('auth',
                    __name__,
                    template_folder='templates',
                    static_folder='static',
                    static_url_path='/auth/static')

ADMIN_ID = 'register1234'

from . import views, functions, event_replay_a, event_replay_b  # noqa: E402, F401, E501
