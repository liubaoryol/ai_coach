'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from gym.envs.registration import register

register(id='envfrommdp-v0',
         entry_point='gym_aicoach.envs:EnvFromMDP',
         max_episode_steps=200)
register(id='envfromcallbacks-v0',
         entry_point='gym_aicoach.envs:EnvFromCallbacks',
         max_episode_steps=200)
register(id='envfromboxpush-v0',
         entry_point='gym_aicoach.envs:EnvFromBoxPushDomain',
         max_episode_steps=200)
