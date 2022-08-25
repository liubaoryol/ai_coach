'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint

exp1_bp = Blueprint('exp1',
                    __name__,
                    template_folder='templates',
                    static_folder='static',
                    static_url_path='/experiment1/static')

from . import views, events_common  # noqa: E402, F401
