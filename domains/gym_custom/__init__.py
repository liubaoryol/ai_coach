'''
Copyright (c) 2020. Sangwon Seo, Vaibhav Unhelkar.
All rights reserved.
'''
from gym.envs.registration import register

register(id='envfrommdp-v0',
         entry_point='gym_custom.envs.mdp_envs:EnvFromMDP',
         max_episode_steps=200)
register(id='envfromcallbacks-v0',
         entry_point='gym_custom.envs.mdp_env:EnvFromCallbacks',
         max_episode_steps=200)
register(id='envfromlatentmdp-v0',
         entry_point='gym_custom.envs.mdp_env:EnvFromLatentMDP',
         max_episode_steps=200)
register(id='envaicoaching-v0',
         entry_point='gym_custom.envs.mdp_env:EnvFromLearnedModels',
         max_episode_steps=200)
register(id='envaicoachingnoop-v0',
         entry_point='gym_custom.envs.mdp_env:EnvFromLearnedModelsNoop',
         max_episode_steps=200)

register(id='circleworld-v0',
         entry_point='gym_custom.envs:CircleWorld',
         max_episode_steps=50)

for idx in range(1, 6):
  register(id=f'MultiGoals2D_{idx}-v0',
           entry_point=f'gym_custom.envs:MultiGoals2D_{idx}',
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

register(id='CleanupSingle-v0',
         entry_point='gym_custom.envs.mdp_env:CleanupSingleEnv_v0',
         max_episode_steps=200)

register(id='EnvMovers-v0',
         entry_point='gym_custom.envs.mdp_env:EnvMovers_v0',
         max_episode_steps=200)

register(id='EnvCleanup-v0',
         entry_point='gym_custom.envs.mdp_env:EnvCleanup_v0',
         max_episode_steps=200)
