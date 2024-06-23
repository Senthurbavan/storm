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
import os
import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import cma

import time
from multiprocessing import Process, Array
import numpy as np
from storm_kit.mpc.task.reacher_task import ReacherTask
np.set_printoptions(precision=2)

recorded_data = torch.load('record_data')

UPPER = np.array([10000, 10000, 500, 100, 500])
LOWER = np.array([1000, 1000, 10, 1, 50])
DEFAULT = np.array([5000, 5000, 100, 15, 100])
PARAMS = ['primitive_collision', 'robot_self_collision',
          'stop_cost', 'goal_pose0', 'goal_pose1']

def transform_params(params):
    params = np.array(params)
    transformed = LOWER + (UPPER - LOWER)*(params/10)
    transformed = transformed.astype('int64')
    param_dict = dict(zip(PARAMS, transformed))
    # print(f'printing param dic.. \n{param_dict}')
    if ('goal_pose0' in param_dict) and ('goal_pose1' in param_dict):
        v = [param_dict['goal_pose0'], param_dict['goal_pose1']]
        del param_dict['goal_pose0']; del param_dict['goal_pose1']
        param_dict['goal_pose'] = v
    # print(f'printing param dic after modification.. \n{param_dict}')
    return param_dict

def inv_transform(param):
    return 10*(param - LOWER)/(UPPER-LOWER)

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

def evaluate(param_list):
    loss = Array('d', [float('inf')]*len(param_list))
    processes = []
    for i in range(len(param_list)):
        param_dic = transform_params(param_list[i])
        p = Process(target=mpc_evaluate, args=(param_dic, i, loss))
        processes.append(p)

    start = time.time()
    for process in processes:
        process.start()

    for process in processes:
        process.join()

    pardur = time.time() - start
    print(f'Execution  time: {pardur:.4f}s')
    print('loss list:')
    [print(f'{x:.7f}', end=', ') for x in np.array(loss)]
    print('')
    return loss


def evaluate_seq(param_list):
    loss = []
    # start = time.time()
    for i in range(len(param_list)):
        param_dic = transform_params(param_list[i])
        # print(f'\n\nparam_dict[{i}] {param_dic}\n')
        error = mpc_evaluate(param_dic, 0)
        loss.append(error)
    # seqdur = time.time() - start
    # print(f'\nExecution  time[{i}]: {seqdur:.4f}s\n')
    return loss

def mpc_evaluate(param_dict, idx=0, loss_list=None):
    robot = 'franka'
    robot_file = robot + '.yml'
    task_file = robot + '_reacher.yml'
    world_file = 'collision_primitives_3d.yml'

    device = torch.device('cuda', 0)
    tensor_args = {'device': device, 'dtype': torch.float32}

    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args, idx=idx)
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
    print(f'cmd_error[{idx}] {cmd_error:.7f}')
    if (loss_list is not None):
        loss_list[idx] = cmd_error
    return cmd_error

if __name__ == '__main__':
    ndim = len(PARAMS)
    # init_param = inv_transform(DEFAULT)
    init_param = np.ones((ndim))*5

    # log_dir = os.environ['SCRATCH']
    log_dir = os.environ['HOME']
    log_file = 'cmaes_log.npy'
    log_file_path = os.path.join(log_dir, log_file)

    stats = {
        'avg_loss': [],
        'theta_mu': [],
        'theta_min_loss': [],
        'results': None
    }

    opts = cma.CMAOptions()
    opts['tolfun'] = 1e-9
    opts['popsize'] = 2#12
    opts['maxiter'] = 2#ndim * 100
    opts['bounds'] = [0, 10]

    es = cma.CMAEvolutionStrategy(init_param, 2, opts)
    i = 0
    t0 = time.time()
    while not es.stop():
        sols = es.ask()
        loss = evaluate(sols)
        loss = np.array(loss)
        idx = np.argmin(loss)
        es.tell(sols, loss)
        es.logger.add()
        es.disp()
        curr_best = np.array(sols).mean(0)
        curr_min = np.array(sols)[idx]
        stats['theta_mu'].append(curr_best)
        stats['avg_loss'].append(loss.mean())
        stats['theta_min_loss'].append(curr_min)
        print("\n\n[INFO] iter %2d | time %10.4f | avg loss %10.7f | min loss %10.7f" % (
            i,
            time.time() - t0,
            loss.mean(), loss.min()))
        V = transform_params(curr_best)
        print(f'current mean parameters: {V}')
        M = transform_params(curr_min)
        print(f'current min-loss parameters: {M}\n\n')
        if (i+1) % 2 == 0:
            with open(log_file_path, 'wb') as f:
                np.save(f, stats)
        i += 1
    es.result_pretty()
    res = es.result
    print(f'\nResults...')
    for r in res:
        print(r)
    stats['results'] = res
    with open(log_file_path, 'wb') as f:
        np.save(f, stats)
    print('\n\n\n========== COMPLETED ==========')