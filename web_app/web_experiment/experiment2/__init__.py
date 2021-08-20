'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint

exp2_bp = Blueprint('exp2',
                    __name__,
                    template_folder='templates',
                    static_folder='static',
                    static_url_path='/experiment2/static')

from . import views, events  # noqa: E402, F401
