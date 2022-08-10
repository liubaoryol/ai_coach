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

from . import views, events_tutorial, events_tutorial2  # noqa: E402, F401
from . import events_exp1_both_tell_align  # noqa: E402, F401
from . import events_exp1_both_user_random  # noqa: E402, F401
from . import events_exp1_both_user_random_2  # noqa: E402, F401
from . import events_exp1_both_user_random_2_intervention
from . import events_exp1_indv_user_random
from . import events_exp1_indv_user_random_2
from . import events_exp1_indv_user_random_2_intervention
from . import events_exp1_indv_tell_align

