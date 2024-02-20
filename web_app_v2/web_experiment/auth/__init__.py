'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint
from web_experiment.define import BPName

auth_bp = Blueprint(BPName.Auth,
                    __name__,
                    template_folder='templates',
                    static_folder='static',
                    static_url_path='/auth/static')

ADMIN_ID = 'tic_web_admin'

from . import views, functions  # noqa: E402, F401, E501
