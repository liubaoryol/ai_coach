'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from gym.envs.registration import register

register(id='circleworld-v0',
         entry_point='gym_custom.envs:CircleWorld',
         max_episode_steps=50)

register(id='MultiGoals2D_1-v0',
         entry_point='gym_custom.envs:MultiGoals2D_1',
         max_episode_steps=200)

register(id='MultiGoals2D_2-v0',
         entry_point='gym_custom.envs:MultiGoals2D_2',
         max_episode_steps=200)

register(id='MultiGoals2D_3-v0',
         entry_point='gym_custom.envs:MultiGoals2D_3',
         max_episode_steps=200)

register(id='AntPush-v0',
         entry_point='gym_custom.envs.ant_maze_env_ex:AntPushEnv_v0',
         max_episode_steps=1000)

register(id='AntPush-v1',
         entry_point='gym_custom.envs.ant_maze_env_ex:AntPushEnv_v1',
         max_episode_steps=1000)

# register(id='AntMaze-v0',
#          entry_point='gym_custom.envs.ant_maze_env_ex:AntMazeEnv_v0',
#          max_episode_steps=1000)

# register(id='AntMaze-v0',
#          entry_point='gym_custom.envs.ant_maze_env_ex:AntMazeEnv_v1',
#          max_episode_steps=1000)

register(id='cleanupsingle-v0',
         entry_point='gym_custom.envs.mdp_env:CleanupSingleEnv_v0',
         max_episode_steps=200)
