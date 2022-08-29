'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint
from web_experiment.define import BPName

replay_bp = Blueprint(BPName.Replay,
                      __name__,
                      template_folder='templates',
                      static_folder='static',
                      static_url_path='/replay/static')

from . import event_replay_record, views
