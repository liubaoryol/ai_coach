'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from gym.envs.registration import register

register(id='envfrommdp-v0',
         entry_point='ai_coach_core.gym.envs:EnvFromMDP',
         max_episode_steps=200)
register(id='envfromcallbacks-v0',
         entry_point='ai_coach_core.gym.envs:EnvFromCallbacks',
         max_episode_steps=200)
register(id='envfromlatentmdp-v0',
         entry_point='ai_coach_core.gym.envs:EnvFromLatentMDP',
         max_episode_steps=200)
