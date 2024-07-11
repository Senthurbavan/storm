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

    # ee_pose_seq = np.load('ee_pos.npy')
    #
    # color = np.array([1.0, 0.0, 0.0])
    # while(True):
    #     try:
    #         gym_instance.step()
    #         gym_instance.clear_lines()
    #         gym_instance.draw_lines(ee_pose_seq, color=color)
    #         if vis_robot: robot_sim.command_robot_position(init_state, env_ptr, robot_ptr)
    #     except KeyboardInterrupt:
    #         print('close')
    #         break

    # ori_ee = np.load('ee_pos_orip2.npy')
    # mod_ee = np.load('ee_pos_mod23.npy')
    #
    # print(f'\n\n\nori shape {ori_ee.shape}')
    # print(f'\n\n\nmod shape {mod_ee.shape}')
    #
    # ee_traj_len = min(ori_ee.shape[0], mod_ee.shape[0])
    #
    # err_L = []
    #
    # for i in range(ee_traj_len):
    #     err = ori_ee[i] - mod_ee[i]
    #     err_L.append(err)
    #
    # loss = np.sum(err_L, axis=0)
    # print(f' error {loss}')


    # ee_traj_seq = np.load('ee_traj_seq_p1.npy', allow_pickle=True)
    # traj1 = ee_traj_seq[0]['ee_pose_seq'].numpy()
    # traj2 = ee_traj_seq[1]['ee_pose_seq'].numpy()
    # # print(type(traj1), type(traj1), traj1.shape, traj2.shape)
    # err_L = []
    # traj_len = min(traj1.shape[0], traj2.shape[0])
    # for i in range(traj_len):
    #     err = traj1[i] - traj2[i]
    #     err = err**2
    #     err_L.append(err)
    #     # [print(f'{e:.7f}', end=' ') for e in err]
    #     # print('')
    # err_mat = np.sum((traj1[:traj_len] - traj2[:traj_len])**2, axis=0)
    #
    # # print(err_L)
    # print(f'traj1:{traj1.shape[0]}, traj2:{traj2.shape[0]}')
    # loss = np.sum(err_L, axis=0)
    # print(f' error {loss}, {err_mat}')
    #
    # color1 = np.array([1.0, 0.0, 0.0])
    # color2 = np.array([0.0, 1.0, 0.0])
    # while(True):
    #     try:
    #         gym_instance.step()
    #         gym_instance.clear_lines()
    #         gym_instance.draw_lines(traj1, color=color1)
    #         gym_instance.draw_lines(traj2, color=color2)
    #         if vis_robot: robot_sim.command_robot_position(init_state, env_ptr, robot_ptr)
    #     except KeyboardInterrupt:
    #         print('close')
    #         break

    # Find all param paths
    p1_files = glob.glob("ee_traj_seq_p1_*")
    p2_files = glob.glob("ee_traj_seq_p2_*")
    p3_files = glob.glob("ee_traj_seq_p3_*")
    p4_files = glob.glob("ee_traj_seq_p4_*")

    print(p4_files)

    # Load and combine the dicts for each param
    ee_trajs_p1_data = []
    ee_trajs_p2_data = []
    ee_trajs_p3_data = []
    ee_trajs_p4_data = []

    for file in p1_files:
        ee_traj_seq = np.load(file, allow_pickle=True)
        ee_trajs_p1_data.extend(ee_traj_seq)

    for file in p2_files:
        ee_traj_seq = np.load(file, allow_pickle=True)
        ee_trajs_p2_data.extend(ee_traj_seq)

    for file in p3_files:
        ee_traj_seq = np.load(file, allow_pickle=True)
        ee_trajs_p3_data.extend(ee_traj_seq)

    for file in p4_files:
        ee_traj_seq = np.load(file, allow_pickle=True)
        ee_trajs_p4_data.extend(ee_traj_seq)

    # Extract the ee_pos and state_seq
    ee_traj_list_p1 = []
    ee_traj_list_p2 = []
    ee_traj_list_p3 = []
    ee_traj_list_p4 = []

    for traj_dict in ee_trajs_p1_data:
        ee_traj_list_p1.append(traj_dict['ee_pose_seq'].numpy())

    for traj_dict in ee_trajs_p2_data:
        ee_traj_list_p2.append(traj_dict['ee_pose_seq'].numpy())

    for traj_dict in ee_trajs_p3_data:
        ee_traj_list_p3.append(traj_dict['ee_pose_seq'].numpy())

    for traj_dict in ee_trajs_p4_data:
        ee_traj_list_p4.append(traj_dict['ee_pose_seq'].numpy())

    print(f'Number of parameter 1 traj: {len(ee_traj_list_p1)}')
    print(f'Number of parameter 2 traj: {len(ee_traj_list_p2)}')
    print(f'Number of parameter 3 traj: {len(ee_traj_list_p3)}')
    print(f'Number of parameter 4 traj: {len(ee_traj_list_p4)}')
    # # Plot the ee_pos traj
    color1 = np.array([0.0, 1.0, 0.0])
    color2 = np.array([1.0, 0.0, 0.0])
    color3 = np.array([0.0, 0.0, 1.0])
    color4 = np.array([0.0, 1.0, 1.0])
    first = True
    while(True):
        try:
            gym_instance.step()
            if first:
                gym_instance.clear_lines()
                for traj1 in ee_traj_list_p1:
                  gym_instance.draw_lines(traj1, color=color1)
                for traj2 in ee_traj_list_p2:
                  gym_instance.draw_lines(traj2, color=color2)
                for traj3 in ee_traj_list_p3:
                  gym_instance.draw_lines(traj3, color=color3)
                for traj4 in ee_traj_list_p4:
                  gym_instance.draw_lines(traj4, color=color4)
                if vis_robot: robot_sim.command_robot_position(init_state, env_ptr, robot_ptr)
            first = False
        except KeyboardInterrupt:
            print('close')
            break


    print('======END=======')
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