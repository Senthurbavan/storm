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

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# import matplotlib
# matplotlib.use('tkagg')

import argparse
import numpy as np
from storm_kit.mpc.task.reacher_task import ReacherTask
np.set_printoptions(precision=2)




def mpc_robot_interactive(args):
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher.yml'
    world_file = 'collision_primitives_3d.yml'

    device = torch.device('cuda', 0)
    tensor_args = {'device': device, 'dtype': torch.float32}

    recorded_data = torch.load('record_data')

    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)

    x_des = recorded_data['x_des']
    mpc_control.update_params(goal_state=x_des)

    sim_dt = mpc_control.exp_params['control_dt']
    assert sim_dt == recorded_data['sim_dt']

    recorded_length = recorded_data['recorded_length'] # the length of the recorded data

    goal_pose_lst = recorded_data['goal_pose']
    t_step_lst = recorded_data['t_step']
    current_robot_state_lst = recorded_data['current_robot_state']
    command_lst = recorded_data['command']

    assert recorded_length == len(goal_pose_lst) == len(t_step_lst) == len(current_robot_state_lst) == len(command_lst)

    print(f'Recorded length: {recorded_length}')
    # for i in range(recorded_length):
        # print(f'\ntime step: {t_step_lst[i]}')
        # print(f'goal pose: {goal_pose_lst[i]}')
        # print(f'curr state')
        # [print(k, current_robot_state_lst[i][k]) for k in current_robot_state_lst[i].keys()]
        # print(f"command: {command_lst[i]['position']}")

    print('Calculating the loop...')
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

        # print(f'i:{i} recorded and computed q_des\n\t{q_des_target}\n\t{q_des}')
        cmd_error = (q_des_target - q_des)
        cmd_error_ratio = abs(cmd_error / q_des_target)*100
        if sum(cmd_error_ratio>5):
            print(f'step: {i} error:', end='\t[')
            [print(f' {e:.2f} ', end='') for e in cmd_error_ratio]
            print(']')
    print(f'Recorded length: {recorded_length}')
    print('End')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    args = parser.parse_args()

    mpc_robot_interactive(args)