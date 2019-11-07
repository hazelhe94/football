from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import gfootball.env as football_env
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()

parser.add_argument('--num-agents', type=int, default=3)
parser.add_argument('--num-policies', type=int, default=3)
parser.add_argument('--num-iters', type=int, default=100000)
parser.add_argument('--simple', action='store_true')



class RllibGFootball(MultiAgentEnv):


  def __init__(self, env_config):
    self.env = football_env.create_environment(
        env_name='test_example_multiagent', stacked=False,
        rewards='scoring',
        #logdir='/tmp/rllib_test',
        enable_goal_videos=False, enable_full_episode_videos=False, render=True,
        dump_frequency=0,
        number_of_left_players_agent_controls=num_agents,
        channel_dimensions=(42, 42))
    self.actions_are_logits = env_config.get("actions_are_logits", False)
    self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
    self.observation_space = gym.spaces.Box(
        low=self.env.observation_space.low[0],
        high=self.env.observation_space.high[0],
        dtype=self.env.observation_space.dtype)
    self.num_agents = num_agents



  def reset(self):
    original_obs = self.env.reset()
    obs = {}
    for x in range(self.num_agents):
      if self.num_agents > 1:
        obs['agent_%d' % x] = original_obs[x]
      else:
        obs['agent_%d' % x] = original_obs
    return obs

  def step(self, action_dict):
    actions = []
    for key, value in sorted(action_dict.items()):
      actions.append(value)
    o, r, d, i = self.env.step(actions)
    rewards = {}
    obs = {}
    for pos, key in enumerate(sorted(action_dict.keys())):
      rewards[key] = r / len(action_dict)
      if self.num_agents > 1:
        obs[key] = o[pos]
      else:
        obs[key] = o
    dones = {'__all__': d}
    infos = i
    return obs, rewards, dones, infos


if __name__ == '__main__':
  args = parser.parse_args()

  
  register_env('gfootball', lambda _: RllibGFootball(args.num_agents))
  single_env = RllibGFootball(args.num_agents)
  obs_space = single_env.observation_space
  act_space = single_env.action_space

  def gen_policy(_):
    return (None, obs_space, act_space, {})


  policies = {
      'policy_{}'.format(i): gen_policy(i) for i in range(args.num_policies)
  }
  policy_ids = list(policies.keys())

  tune.run(
      'contrib/MADDPG',
      stop={'training_iteration': args.num_iters},
      checkpoint_freq=50,
      config={
          'env': 'gfootball',
          'learning_starts': 100,
           'env_config': {
                'actions_are_logits': True,
            },
          'train_batch_size': 2000,
          'multiagent': {
              'policies': policies,
              'policy_mapping_fn': tune.function(
                  lambda agent_id: policy_ids[int(agent_id[6:])]),
          },
      },
  )
