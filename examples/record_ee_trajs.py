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


def mpc_robot_interactive(args, gym_instance, seed_val=0):
    vis_ee_target = False
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
    env_ptr = gym_instance.env_list[0]
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    device = torch.device('cuda', 0)
    tensor_args = {'device': device, 'dtype': torch.float32}

    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)

    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)

    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args, sd=seed_val)
    # param 1
    p1 = {'goal_pose': {'weight':[15.0, 1500.0]}, 'primitive_collision': {'weight':500.0},
          'manipulability': {'weight':30.0}, 'stop_cost': {'weight':150.0}}
    p2 = {'goal_pose': {'weight': [5.0, 100.0]}, 'primitive_collision': {'weight': 10000.0},
          'manipulability': {'weight': 0.10}, 'stop_cost': {'weight': 10.0}}
    mpc_control.controller.rollout_fn.change_cost_params(p2)

    # Set the Target Pose
    x_pos = np.array([0.0, 0.0, 0.0])
    x_q = np.array([0.0, 0.0, 0.0, 0.0])
    target_mug_pose = gymapi.Transform()
    target_mug_pose.p = gymapi.Vec3(-0.65, 1.25, 0.1)
    target_mug_pose.r = gymapi.Quat(0.7071, 0.0, 0.0, 0.7071)
    target_mug_pose = copy.deepcopy(w_T_r.inverse() * target_mug_pose)

    x_pos[0] = target_mug_pose.p.x
    x_pos[1] = target_mug_pose.p.y
    x_pos[2] = target_mug_pose.p.z
    x_q[1] = target_mug_pose.r.x
    x_q[2] = target_mug_pose.r.y
    x_q[3] = target_mug_pose.r.z
    x_q[0] = target_mug_pose.r.w

    mpc_control.update_params(goal_ee_pos=x_pos, goal_ee_quat=x_q)

    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w, w_T_r.r.x, w_T_r.r.y, w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0, 3] = w_T_r.p.x
    w_T_robot[1, 3] = w_T_r.p.y
    w_T_robot[2, 3] = w_T_r.p.z
    w_T_robot[:3, :3] = rot[0]

    w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3, 3].unsqueeze(0),
                                        rot=w_T_robot[0:3, 0:3].unsqueeze(0))

    sim_dt = mpc_control.exp_params['control_dt']
    t_step = gym_instance.get_sim_time()

    ee_pose_seq = []
    last_ee_pose = None
    lase_ee_pose_update = time.time()
    sim_start_time = time.time()
    i = 0
    while (i > -100):
        try:
            if (time.time() - sim_start_time) > 210:
                print('\n\n Simulation is taking too long .. Stopping...')
                break
            if(time.time() - lase_ee_pose_update) > 30:
                print('\n\n Robot is not moving.. Stopping...')
                break
            gym_instance.step()
            # if(i==0): input("\nPress Enter...\n")
            t_step += sim_dt

            current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))

            command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
            q_des = copy.deepcopy(command['position'])

            curr_state = np.hstack((current_robot_state['position'], current_robot_state['velocity'], current_robot_state['acceleration']))
            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            ee_pose_seq.append(copy.deepcopy(e_pos))

            dist = 0
            if last_ee_pose is not None:
                dist = np.linalg.norm(e_pos - last_ee_pose)
            if (last_ee_pose is None) or dist > 0.025:
                print(f'Updating dist{dist:.7f}')
                last_ee_pose = e_pos
                lase_ee_pose_update = time.time()

            robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
            i+=1

        except KeyboardInterrupt:
            print('Closing')
            break

    ee_pose_seq = torch.tensor(ee_pose_seq).to('cpu')
    ee_pose_seq = w_robot_coord.transform_point(ee_pose_seq)

    mpc_control.close()
    return ee_pose_seq


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
    # gym_instance = Gym(**sim_params)

    # ee_traj = mpc_robot_interactive(args, gym_instance, seed_val=238)
    # save the data
    # with open('ee_pos_mod23.npy', 'wb') as f:
    # np.save(f, ee_traj)

    seed_val_list = [17, 8]
    ee_traj_seq = []
    gym_instance = Gym(**sim_params)
    for i in range(1):
        print(f'Iteration {i+1}, seed value {seed_val_list[i]}')
        ee_traj = mpc_robot_interactive(args, gym_instance, seed_val=seed_val_list[i])
        ee_traj_seq.append(ee_traj)
        print(f'Traj length {ee_traj.shape[0]}')
        # del gym_instance

    # save the data
    with open('ee_traj_seq.npy', 'wb') as f:
        np.save(f, ee_traj_seq)