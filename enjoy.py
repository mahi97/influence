import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from influence.envs import VecPyTorch, make_vec_envs
from influence.utils import get_render_func, get_vec_normalize

if __name__ == '__main__':
    sys.path.append('influence')

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--env-name',
        default='MarlGrid-PassDoor30x30-v1',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--load-dir',
        default='./trained_models/ppo',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')

    args = parser.parse_args()

    args.det = not args.non_det

    env = make_vec_envs(
        args.env_name,
        args.seed + 1000,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=False)

    # Get a render function
    render_func = get_render_func(env)

    # We need to use the same statistics for normalization as used in training
    actor_critic_and_obs_rms = [torch.load(os.path.join(args.load_dir, args.env_name + '{}_agent_{}_'.format(args.seed, i) + ".pt"),
                                           map_location='cpu') for i in range(2)]
    actor_critics = [a[0] for a in actor_critic_and_obs_rms]
    obs_rmss = [a[1] for a in actor_critic_and_obs_rms]
    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rmss[0]

    recurrent_hidden_states = [torch.zeros(1, ac.recurrent_hidden_state_size) for ac in actor_critics]
    masks = torch.zeros(1, 1)

    obs = env.reset()

    if render_func is not None:
        render_func('human')

    if args.env_name.find('Bullet') > -1:
        import pybullet as p

        torsoId = -1
        for i in range(p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == "torso"):
                torsoId = i

    while True:
        with torch.no_grad():
            actions = []
            temp_rhs = []
            for ac, rhs in zip(actor_critics, recurrent_hidden_states):
                _, action, _, rhs = ac.act(obs, rhs, masks, deterministic=args.det)
                actions.append(action)
                temp_rhs.append(rhs)
            recurrent_hidden_states = temp_rhs
        # Observe reward and next obs
        obs, reward, done, _ = env.step(torch.stack(actions).view(1, 2, 1))

        masks.fill_(0.0 if done else 1.0)

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            render_func('human')
