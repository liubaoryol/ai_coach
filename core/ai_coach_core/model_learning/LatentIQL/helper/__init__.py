'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''

from ai_coach_core.model_learning.IQLearn.utils.logger import AGENT_TRAIN_FORMAT

AGENT_TRAIN_FORMAT['miql'] = [
    # ('batch_reward', 'BR', 'float'),
    ('actor_loss', 'ALOSS', 'float'),
    ('critic_loss', 'CLOSS', 'float'),
    ('alpha_loss', 'TLOSS', 'float'),
    ('alpha_value', 'TVAL', 'float'),
    ('actor_entropy', 'AENT', 'float')
]

AGENT_TRAIN_FORMAT['msac'] = [
    # ('batch_reward', 'BR', 'float'),
    ('actor_loss', 'ALOSS', 'float'),
    ('critic_loss', 'CLOSS', 'float'),
    ('alpha_loss', 'TLOSS', 'float'),
    ('alpha_value', 'TVAL', 'float'),
    ('actor_entropy', 'AENT', 'float')
]
