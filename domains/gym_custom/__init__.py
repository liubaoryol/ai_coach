'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from gym.envs.registration import register

register(id='circleworld-v0',
         entry_point='gym_custom.envs:CircleWorld',
         max_episode_steps=50)
