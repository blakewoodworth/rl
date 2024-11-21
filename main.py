#!/usr/bin/env python 

import torch
import gymnasium as gym
import numpy as np
import argparse
from tqdm import tqdm
import wandb

# def get_device():
# 	return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_args():
	argparser = argparse.ArgumentParser(description='My PPO implementation')

	argparser.add_argument('--env_name', type=str, default='LunarLander-v3', help='Choose the gym environment')
	argparser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden dims in agent network')
	argparser.add_argument('--time_steps_per_batch', type=int, default=750, help='Number of time steps per training batch')
	argparser.add_argument('--max_time_steps_per_episode', type=int, default=250, help='Max number of time steps per episode')
	argparser.add_argument('--n_updates', type=int, default=1500, help='Max number of PPO updates')
	argparser.add_argument('--inner_epochs', type=int, default=3, help='Max number of inner epochs per PPO update')
	argparser.add_argument('--batch_size', type=int, default=250, help='Batch size for inner loop per PPO update')
	argparser.add_argument('--clip_epsilon', type=float, default=0.2, help='Clip epsilon parameter for PPO')
	argparser.add_argument('--value_coefficient', type=float, default=0.5, help='Weight on the value term in PPO objective')
	argparser.add_argument('--entropy_coefficient', type=float, default=0.01, help='Weight on the entropy term in PPO objective')
	argparser.add_argument('--gamma', type=float, default=0.999, help='Discount factor')
	argparser.add_argument('--gae_lambda', type=float, default=0.999, help='Generalized advantage estimator lambda value')
	argparser.add_argument('--stepsize', type=float, default=2.5e-4, help='Generalized advantage estimator lambda value')
	argparser.add_argument('--stepsize_anneal_factor', type=float, default=1e-2, help='Generalized advantage estimator lambda value')
	return vars(argparser.parse_args())

class Agent(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.actor = torch.nn.Sequential(
			torch.nn.Linear(np.array(args['observation_shape']).prod(), args['hidden_dim']),
			torch.nn.GELU(),
			torch.nn.Linear(args['hidden_dim'], args['hidden_dim']),
			torch.nn.GELU(),
			torch.nn.Linear(args['hidden_dim'], args['action_n'])
		)
		self.critic = torch.nn.Sequential(
			torch.nn.Linear(np.array(args['observation_shape']).prod(), args['hidden_dim']),
			torch.nn.GELU(),
			torch.nn.Linear(args['hidden_dim'], args['hidden_dim']),
			torch.nn.GELU(),
			torch.nn.Linear(args['hidden_dim'], 1)
		)

	def action_and_value(self, obs, action=None):
		probs = torch.distributions.Categorical(logits=self.actor(obs))
		if action is None:
			action = probs.sample()
		return action, probs.log_prob(action), probs.entropy(), self.critic(obs)

def generalized_advantage_estimate(rewards, values, g, l):
	deltas = np.array(rewards) - np.array(values)
	deltas[:-1] += g*np.array(values)[1:]

	backward_gaes = [deltas[-1]]
	for i in reversed(range(len(rewards)-1)):
		backward_gaes.append(deltas[i] + g*l*backward_gaes[-1])
	backward_gaes.reverse()
	backward_gaes = torch.Tensor(backward_gaes)
	return (backward_gaes - backward_gaes.mean())/backward_gaes.std()

def collect_trajectories(env, agent, args):
	# device = get_device()
	observations = []
	actions = []
	reward_to_gos = []
	logprobs = []
	advantages = []

	total_reward = 0.
	n_episodes = 0.
	
	t = 0
	while t < args['time_steps_per_batch']:
		n_episodes += 1
		total_episode_reward = 0.
		episode_t = 0
		done = False
		episode_rewards = []
		episode_values = []
		obs, _ = env.reset()
		while not done:
			episode_t += 1

			observations.append(torch.Tensor(obs))

			with torch.no_grad():
				action, logprob, _, value = agent.action_and_value(observations[-1])
			actions.append(action)
			logprobs.append(logprob)
			episode_values.append(value.item())
			
			obs, reward, terminated, truncated, _ = env.step(action.numpy())

			total_episode_reward += reward
			episode_rewards.append(reward)

			done = (episode_t > args['max_time_steps_per_episode']) or terminated or truncated
		
		total_reward += total_episode_reward
		reward_to_gos.append(torch.Tensor(np.cumsum(episode_rewards[::-1])[::-1].copy()))
		advantages.append(generalized_advantage_estimate(episode_rewards, episode_values, args['gamma'], args['gae_lambda']))

		t += episode_t

	observations = torch.stack(observations)
	actions = torch.Tensor(actions)
	reward_to_gos = torch.Tensor(torch.cat(reward_to_gos))
	logprobs = torch.Tensor(logprobs)
	advantages = torch.Tensor(torch.cat(advantages))

	return observations, actions, reward_to_gos, logprobs, advantages, total_reward/n_episodes

def ppo_clip(env, agent, args):

	optimizer = torch.optim.Adam(agent.parameters(), lr=args['stepsize'])
	verb = 'deprecated' if int(torch.__version__[0])>1 else False
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['n_updates'], eta_min=args['stepsize_anneal_factor']*args['stepsize'], verbose=verb)

	for i_update in range(1,args['n_updates']+1):

		# collect trajectories and reward to gos
		observations, actions, reward_to_gos, logprobs, advantages, avg_reward = collect_trajectories(env, agent, args)
		n_times = observations.shape[0]

		wandb.log({'avg_reward': avg_reward}, commit=False)
		wandb.log({'stepsize': optimizer.param_groups[0]['lr']})

		if i_update % 50 == 0:
			print(f"Update {i_update}/{args['n_updates']}: average reward {avg_reward:.3f}")

		# optimize PPO-clip objective, value function, and entropy loss
		for inner_epoch in range(args['inner_epochs']):
			
			batch_order = np.random.permutation(n_times-(n_times%args['batch_size']))

			for i in range(0,len(batch_order),args['batch_size']):
				# get batch
				start, end = i, i+args['batch_size']
				batch_idxs = batch_order[start:end]

				batch_obs = observations[batch_idxs]
				batch_actions = actions[batch_idxs]
				batch_rtg = reward_to_gos[batch_idxs]
				batch_logprob = logprobs[batch_idxs]
				batch_advantage = advantages[batch_idxs]

				# calculate ppo-clip loss
				_, new_logprob, new_entropy, new_value = agent.action_and_value(batch_obs, batch_actions)
				ratio = (new_logprob - batch_logprob).exp()

				ppo_clip_loss1 = -ratio * batch_advantage
				ppo_clip_loss2 = -torch.clamp(ratio, 1-args['clip_epsilon'], 1+args['clip_epsilon'])
				ppo_clip_loss = torch.max(ppo_clip_loss1, ppo_clip_loss2).mean()

				# calculate value loss
				value_loss = ((new_value - batch_rtg)**2).mean()

				# calculate entropy loss
				entropy_loss = -new_entropy.mean()

				wandb.log({'losses/ppo_clip_loss': ppo_clip_loss}, commit=False)
				wandb.log({'losses/value_loss': value_loss}, commit=False)
				wandb.log({'losses/entropy_loss': entropy_loss}, commit=False)

				all_loss = ppo_clip_loss + args['value_coefficient']*value_loss + args['entropy_coefficient']*entropy_loss

				optimizer.zero_grad()
				all_loss.backward()
				optimizer.step()
		scheduler.step()

if __name__ == '__main__':
	np.random.seed(1)
	torch.manual_seed(1)

	args = parse_args()

	wandb.init(project='rl-ppo', config=args)

	env = gym.make(args['env_name'])
	assert isinstance(env.action_space, gym.spaces.Discrete), 'this code only works for discrete actions'

	args['observation_shape'] = env.observation_space.shape
	args['action_n'] = env.action_space.n

	agent = Agent(args)

	ppo_clip(env, agent, args)

	env = gym.make(args['env_name'], render_mode='rgb_array')
	env = gym.wrappers.RecordVideo(env, f"videos/{args['env_name']}")
	
	obs, _ = env.reset()
	done = False
	rewards = 0.
	while not done:
		with torch.no_grad():
			action = agent.action_and_value(torch.Tensor(obs))[0]
		obs, reward, terminated, truncated, _ = env.step(action.numpy())
		rewards += reward
		done = terminated or truncated
	print(f'Final simulation total reward: {rewards}')
	env.close()
	wandb.finish()

