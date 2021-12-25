'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from gym.envs.registration import register

register(
    id='envfrommdp-v0',
    entry_point='gym_aicoach.envs:GymEnvFromMDP',
)
