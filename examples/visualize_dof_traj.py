import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Find all param paths
p1_files = glob.glob("ee_traj_seq_p1_*")
p2_files = glob.glob("ee_traj_seq_p2_*")

# Load and combine the dicts for each param
ee_trajs_p1_data = []
ee_trajs_p2_data = []

for file in p1_files:
    ee_traj_seq = np.load(file, allow_pickle=True)
    ee_trajs_p1_data.extend(ee_traj_seq)

for file in p2_files:
    ee_traj_seq = np.load(file, allow_pickle=True)
    ee_trajs_p2_data.extend(ee_traj_seq)

# Extract the state_seq
dof_state_list_p1 = []
dof_state_list_p2 = []

for traj_dict in ee_trajs_p1_data:
    dof_state_traj = traj_dict['robot_state_seq']
    dof_traj = []
    for point_dic in dof_state_traj:
        dof_traj.append(point_dic['position'])
    dof_traj = np.array(dof_traj)
    dof_state_list_p1.append(dof_traj)

for traj_dict in ee_trajs_p2_data:
    dof_state_traj = traj_dict['robot_state_seq']
    dof_traj = []
    for point_dic in dof_state_traj:
        dof_traj.append(point_dic['position'])
    dof_traj = np.array(dof_traj)
    dof_state_list_p2.append(dof_traj)

print(len(dof_state_list_p1))
print(dof_state_list_p1[0])
print(dof_state_list_p1[0].shape)

print(len(dof_state_list_p2))
print(dof_state_list_p2[0])
print(dof_state_list_p2[0].shape)

# fig, axes = plt.subplots(7, 1, figsize=(10, 15))
# num_trials = len(dof_state_list_p1)
# for i in range(7):
#     for trial in range(num_trials):
#         axes[i].plot(range(len(dof_state_list_p1[trial])),
#                      dof_state_list_p1[trial][:,i], color='red')
#         axes[i].plot(range(len(dof_state_list_p2[trial])),
#                      dof_state_list_p2[trial][:, i], color='blue')
#     axes[i].set_title(f'DOF {i+1} Trajectories')
#     axes[i].set_xlabel('Steps')
#
# plt.show()

# num_trials = len(dof_state_list_p1)
# for i in range(7):
#     plt.figure()
#     for trial in range(num_trials):
#         plt.plot(range(len(dof_state_list_p1[trial])),
#                      dof_state_list_p1[trial][:,i], color='red')
#         plt.plot(range(len(dof_state_list_p2[trial])),
#                  dof_state_list_p2[trial][:, i], color='blue')
#     plt.title(f'Joint {i+1} Angles')
#     plt.xlabel('Time')
# plt.show()

num_trials = len(dof_state_list_p1)
for i in range(7):
    fig, axes = plt.subplots(3, 1)
    for trial in range(num_trials):
        axes[0].plot(range(len(dof_state_list_p1[trial])),
                     dof_state_list_p1[trial][:, i], color='red')
        axes[0].set_title(f'DOF {i+1} Trajectory for Parameter 1')
        axes[1].plot(range(len(dof_state_list_p2[trial])),
                     dof_state_list_p2[trial][:, i], color='blue')
        axes[1].set_title(f'DOF {i+1} Trajectory for Parameter 2')
        axes[1].set_ylabel('Joint Angle (Radians)')
        axes[2].plot(range(len(dof_state_list_p1[trial])),
                     dof_state_list_p1[trial][:,i], color='red')
        axes[2].plot(range(len(dof_state_list_p2[trial])),
                 dof_state_list_p2[trial][:, i], color='blue')
        axes[2].set_title('Both in Same Plot')
        axes[2].set_xlabel('Time Steps')
plt.tight_layout()
plt.show()