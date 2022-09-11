'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint

demo_bp = Blueprint("demo",
                    __name__,
                    template_folder='templates',
                    static_folder='static',
                    static_url_path='/demo/static')

from . import views  # noqa: E402, F401
from . import events  # noqa: E402, F401
