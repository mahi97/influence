import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from influence import algo, utils
from influence.arguments import get_args
from influence.envs import make_vec_envs
from influence.model import Policy
from influence.storage import RolloutStorage
from evaluation import evaluate

import wandb
import matplotlib.pyplot as plt


def main():
    args = get_args()
    wandb.init(project='influence')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    actor_critics = [Policy(
        envs.observation_space.shape,
        act,
        base_kwargs={'recurrent': args.recurrent_policy}) for act in envs.action_space]

    [ac.to(device) for ac in actor_critics]

    if args.algo == 'ppo':
        agents = [algo.PPO(
            ac,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm) for ac in actor_critics]

    rolloutss = [RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space[i],
                                ac.recurrent_hidden_state_size) for i, ac in enumerate(actor_critics)]

    obs = envs.reset()
    for rollouts in rolloutss:
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            [utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr) for agent in agents]

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                values = []
                actions = []
                action_log_probs = []
                recurrent_hidden_statess = []
                for ac, rollouts in zip(actor_critics, rolloutss):
                    value, action, action_log_prob, recurrent_hidden_states = ac.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                    values.append(value)
                    actions.append(action)
                    action_log_probs.append(action_log_prob)
                    recurrent_hidden_statess.append(recurrent_hidden_states)

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(torch.stack(actions).view(args.num_processes, 2, 1).to(device))

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            for rollouts, action, value, recurrent_hidden_states, action_log_prob in zip(rolloutss, actions, values,
                                                                                         recurrent_hidden_statess,
                                                                                         action_log_probs):
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

        for i, (agent, ac, rollouts) in enumerate(zip(agents, actor_critics, rolloutss)):
            with torch.no_grad():
                next_value = ac.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.gamma, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass
                torch.save([
                    ac,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + '{}_agent_{}_'.format(args.seed, i) + ".pt"))

            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "Agent {} Updates {}, num timesteps {}, FPS {} \n"
                    "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                        .format(i, j, total_num_steps,
                                int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards),
                                np.median(episode_rewards), np.min(episode_rewards),
                                np.max(episode_rewards), dist_entropy, value_loss,
                                action_loss))
                if i == 0:
                    # plt.imshow(np.sum(rollouts.counter, axis=(2, 3, 4, 5, 6)).T)
                    plt.imshow(np.sum(rollouts.dcounter, axis=(2, 3, 4, 5)).T)
                    # plt.imshow(np.sum(rollouts.counter[:, :, :, :, :, :, 2], axis=(2, 3, 4, 5)).T)
                else:
                    # plt.imshow(np.sum(rollouts.counter, axis=(0, 1, 2, 5, 6)).T)
                    plt.imshow(np.sum(rollouts.dcounter, axis=(0, 1, 2, 5)).T)
                    # plt.imshow(np.sum(rollouts.counter[:, :, :, :, :, :, 2], axis=(2, 3, 4, 5)).T)

                i = str(i)
                wandb.log({
                    'FPS ' + i: int(total_num_steps / (end - start)),
                    'Mean Reward ' + i: np.mean(episode_rewards),
                    'Median Reward ' + i: np.mean(episode_rewards),
                    'Median Reward ' + i: np.median(episode_rewards),
                    'Max Reward ' + i: np.max(episode_rewards),
                    'Min Reward ' + i: np.min(episode_rewards),
                    'Dist Entropy ' + i: dist_entropy,
                    'Value Loss ' + i: value_loss,
                    'Action Loss ' + i: action_loss,
                    'Chart ' + i: plt
                })
                # plt.show()

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critics, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
