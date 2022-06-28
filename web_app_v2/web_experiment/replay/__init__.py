'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint

replay_bp = Blueprint('replay',
                    __name__,
                    template_folder='templates',
                    static_folder='static',
                    static_url_path='/replay/static')

from . import views, event_replay_a_record, event_replay_b_record