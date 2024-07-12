#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
""" Example spawning a robot in gym

"""
import copy
from isaacgym import gymapi
from isaacgym import gymutil

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#

import os
import glob

import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
np.set_printoptions(precision=2)


def mpc_robot_interactive(args, gym_instance):
    vis_ee_target = True
    vis_robot = False
    gym = gym_instance.gym
    sim = gym_instance.sim

    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher.yml'
    world_file = 'collision_primitives_3d.yml'

    world_yml = join_path(get_gym_configs_path(), world_file)
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)

    robot_yml = join_path(get_gym_configs_path(), args.robot + '.yml')
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    if (args.cuda):
        device = 'cuda'
    else:
        device = 'cpu'

    sim_params['collision_model'] = None

    # create robot simulation:
    robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, device=device)

    # create gym environment:
    robot_pose = sim_params['robot_pose']
    init_state = sim_params['init_state']
    env_ptr = gym_instance.env_list[0]
    if vis_robot:
        robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    p = gymapi.Vec3(robot_pose[0], robot_pose[1], robot_pose[2])
    robot_pose = gymapi.Transform(p=p, r=gymapi.Quat(robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6]))

    w_T_r = copy.deepcopy(robot_pose)

    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)

    recorded_data = np.load('record_data_loss_test1.npy', allow_pickle=True)
    recorded_data = recorded_data[0]

    play_data = np.load('multi_seed_traj_output.npy', allow_pickle=True)
    play_data = play_data[0]

    ee_traj = recorded_data['ee_pose_seq']
    print(f"ee_traj len: {len(ee_traj)}")
    command_ee_traj = recorded_data['command_ee_pose_seq']
    recorded_w_T_r = recorded_data['w_T_r']

    play_ee_traj = play_data['ee_pose_seq']
    print(f"play_ee_traj len: {len(play_ee_traj)}")
    command_play_ee_traj = play_data['command_ee_pose_seq']

    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w, w_T_r.r.x, w_T_r.r.y, w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0, 3] = w_T_r.p.x
    w_T_robot[1, 3] = w_T_r.p.y
    w_T_robot[2, 3] = w_T_r.p.z
    w_T_robot[:3, :3] = rot[0]

    w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3, 3].unsqueeze(0),
                                        rot=w_T_robot[0:3, 0:3].unsqueeze(0))

    print(f'w_T_r: \n {w_T_r.p.x}, {w_T_r.p.y}, {w_T_r.p.z}, '
          f'{w_T_r.r.x}, {w_T_r.r.y}, {w_T_r.r.z}, {w_T_r.r.w}')
    print(f'recorded w_T_r: \n{recorded_w_T_r}')

    ee_pos_seq = np.vstack(np.array(ee_traj)[:, 0])
    ee_pos_seq = torch.tensor(ee_pos_seq).to('cpu')
    ee_pos_seq = w_robot_coord.transform_point(ee_pos_seq)

    command_ee_pos_seq = np.vstack(np.array(command_ee_traj)[:, 0])
    command_ee_pos_seq = torch.tensor(command_ee_pos_seq).to('cpu')
    command_ee_pos_seq = w_robot_coord.transform_point(command_ee_pos_seq)

    command_delta = torch.stack((ee_pos_seq, command_ee_pos_seq), dim=1)

    play_ee_pos_seq = np.vstack(np.array(play_ee_traj)[:, 0])
    play_ee_pos_seq = torch.tensor(play_ee_pos_seq).to('cpu')
    play_ee_pos_seq = w_robot_coord.transform_point(play_ee_pos_seq)

    command_play_ee_pos_seq = np.vstack(np.array(command_play_ee_traj)[:, 0])
    command_play_ee_pos_seq = torch.tensor(command_play_ee_pos_seq).to('cpu')
    command_play_ee_pos_seq = w_robot_coord.transform_point(command_play_ee_pos_seq)

    command_play_delta = torch.stack((play_ee_pos_seq, command_play_ee_pos_seq), dim=1)

    er = torch.sum((ee_pos_seq - play_ee_pos_seq)**2)
    print(f'\n\nThe error is {er} \n\n')

    print(f'ee_pos_seq: {ee_pos_seq.shape}, '
          f'command_ee_pos_seq: {command_ee_pos_seq.shape}, '
          f'command delta: {command_delta.shape}, '
          f'play_ee_pos_seq: {play_ee_pos_seq.shape}')

    color1 = np.array([1.0, 0.0, 0.0])
    color2 = np.array([0.0, 1.0, 0.0])
    color3 = np.array([0.0, 0.0, 1.0])
    color_list = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    first = True
    while(True):
        try:
            gym_instance.step()
            if first:
                first = False
                gym_instance.clear_lines()
                gym_instance.draw_lines(ee_pos_seq, color=color1)
                # gym_instance.draw_lines(play_ee_pos_seq, color=color2)
                # gym_instance.draw_lines(command_ee_pos_seq, color=color2)
                for j in range(len(command_delta)):
                    if j%15 == 0:
                        gym_instance.draw_lines(command_delta[j], color=color_list[1])
                        gym_instance.draw_lines(command_play_delta[j], color=color_list[2])
        except KeyboardInterrupt:
            print('close')
            break

    print('===End===')
    return 1


if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()

    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)

    mpc_robot_interactive(args, gym_instance)