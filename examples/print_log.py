import numpy as np
import os
import cma

UPPER = np.array([10000, 10000, 500, 100, 500])
LOWER = np.array([1000, 1000, 10, 1, 50])
DEFAULT = np.array([5000, 5000, 100, 15, 100])
PARAMS = ['primitive_collision', 'robot_self_collision',
          'stop_cost', 'goal_pose0', 'goal_pose1']

def transform_params(params):
    params = np.array(params)
    transformed = LOWER + (UPPER - LOWER)*(params/10)
    transformed = transformed.astype('int64')
    return transformed

# log_dir = os.environ['SCRATCH']
log_dir = os.environ['HOME']
log_file = 'cmaes_log.npy'
log_file_path = os.path.join(log_dir, log_file)

log = np.load(log_file_path, allow_pickle=True)

avg_loss = log.item().get('avg_loss')
theta_mu = log.item().get('theta_mu')
theta_min_loss = log.item().get('theta_min_loss')
res = log.item().get('results')

print(f'Total iterations: {len(avg_loss)}')
print('Parameters: ', PARAMS)
print('Default Parameters: ', DEFAULT)
print('Printing iteration...')
for i in range(len(theta_mu)):
    print(f'{i:5d}: ', end=' ')
    print(transform_params(theta_mu[i]), end='\t')
    print(f'Avg loss: {avg_loss[i]:.7f}')

print('(Raw parameters)Printing iteration...')
for i in range(len(theta_mu)):
    print(f'{i:5d}: ', end=' ')
    print(theta_mu[i], end='\t')
    print(f'Avg loss: {avg_loss[i]:.7f}')

print(f'printing results...')
for r in res:
    print(r)
print('Result Params...')
print(transform_params(res[5]))