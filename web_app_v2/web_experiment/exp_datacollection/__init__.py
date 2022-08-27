'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint
from web_experiment.define import BPName

exp_dcollect_bp = Blueprint(BPName.Exp_datacol,
                            __name__,
                            template_folder='templates',
                            static_folder='static',
                            static_url_path='/exp_datacollection/static')

from . import views  # noqa: E402, F401
from . import events  # noqa: E402, F401
