'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from flask import Blueprint
from web_experiment.define import BPName

consent_bp = Blueprint(BPName.Consent,
                       __name__,
                       template_folder='templates',
                       static_folder='static',
                       static_url_path='/consent/static')

from . import views  # noqa: E402, F401
