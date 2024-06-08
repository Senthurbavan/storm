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



import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import time
from multiprocessing import Process, Array, Pool
import concurrent.futures

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

recorded_data = torch.load('record_data')

def process_arguments(n, param_list):
    for i in range(n):
        yield param_list[i]

def change_params(mpc_c, param_dict):
    if 'horizon' in param_dict.keys():
        mpc_c.change_horizon(param_dict['horizon'])
        param_dict.pop('horizon')

    if 'num_particles' in param_dict.keys():
        mpc_c.change_num_particles(param_dict['num_particles'])
        param_dict.pop('num_particles')

    # change cost weights
    cost_params = {}
    for k, v in param_dict.items():
        cost_params[k] = {'weight':v}
    mpc_c.controller.rollout_fn.change_cost_params(cost_params)

def mpc_evaluate(param_dict):

    robot = 'franka'
    robot_file = robot + '.yml'
    task_file = robot + '_reacher.yml'
    world_file = 'collision_primitives_3d.yml'

    device = torch.device('cuda', 0)
    tensor_args = {'device': device, 'dtype': torch.float32}

    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
    change_params(mpc_control, param_dict)

    x_des = recorded_data['x_des']
    mpc_control.update_params(goal_state=x_des)

    sim_dt = mpc_control.exp_params['control_dt']
    assert sim_dt == recorded_data['sim_dt']

    recorded_length = recorded_data['recorded_length']  # the length of the recorded data
    goal_pose_lst = recorded_data['goal_pose']
    t_step_lst = recorded_data['t_step']
    current_robot_state_lst = recorded_data['current_robot_state']
    command_lst = recorded_data['command']

    assert recorded_length == len(goal_pose_lst) == len(t_step_lst) == len(current_robot_state_lst) == len(command_lst)

    dof = command_lst[0]['position'].shape[0]
    cmd_error = np.zeros(dof)

    for i in range(recorded_length):
        goal_pose = goal_pose_lst[i]
        if goal_pose is not None:
            g_pos, g_q = goal_pose
            mpc_control.update_params(goal_ee_pos=g_pos,
                                      goal_ee_quat=g_q)

        t_step = t_step_lst[i]
        current_robot_state = current_robot_state_lst[i]

        command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
        q_des = copy.deepcopy(command['position'])
        q_des_target = copy.deepcopy((command_lst[i])['position'])
        cmd_error += (q_des_target - q_des)**2

    mpc_control.close()

    cmd_error /= recorded_length
    cmd_error = np.sum(cmd_error)/dof
    print(f'cmd_error {cmd_error:.4f}')
    return cmd_error
    # res_array[res_array_index] = cmd_error
    return

if __name__ == '__main__':

    # ===== Only one =====
    # start = time.time()
    # mpc_evaluate({'horizon':40, 'state_bound':1})
    # dur = time.time() - start
    # print(f'Execution  time: {dur:.4f}s')

    # ==== Multiple processes - Manual ====
    # num_process = 4
    # res = Array('d', [0]*num_process)
    # param_l = [{'horizon':40, 'state_bound':1},
    #            {'horizon':50, 'state_bound':100},
    #            {'horizon': 45, 'state_bound': 1},
    #            {'horizon': 55, 'state_bound': 100}]
    #
    # p0 = Process(target=mpc_evaluate, args=(args, param_l[0], res, 0))
    # p1 = Process(target=mpc_evaluate, args=(args, param_l[1], res, 1))
    #
    # start = time.time()
    # p0.start()
    # p1.start()
    #
    # p0.join()
    # p1.join()

    # dur = time.time() - start
    # print(f'Execution  time: {dur:.4f}s')

    # ==== Multiple processes ====
    num_process = 2
    param_l = [{'horizon':40, 'state_bound':1},
               # {'horizon':50, 'state_bound':100},
               # {'horizon': 45, 'state_bound': 1},
               {'horizon': 55, 'state_bound': 100}]

    processes = []
    process_args = process_arguments(num_process, param_l)
    for k in param_l:
        print(f'\n%%%%%%%%arg:{k}--{type(k)}')
        p = Process(target=mpc_evaluate, args=(k,))
        processes.append(p)

    start = time.time()
    for process in processes:
        process.start()

    for process in processes:
        process.join()

    pardur = time.time() - start
    print(f'Execution  time: {pardur:.4f}s')

    #
    # print('\n============================\nseq execution....')
    # start = time.time()
    # for i in range(num_process):
    #     mpc_evaluate(args, param_l[i], res, i)
    # seqdur = time.time() - start
    #
    # print('Results')
    # for r in res:
    #     print(f'{r:.4f}')
    #
    # print(f'Par Execution  time: {pardur:.4f}s')
    # print(f'Seq Execution  time: {seqdur:.4f}s')
