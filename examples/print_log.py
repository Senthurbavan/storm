import numpy as np
import os
import cma

UPPER = np.array([10000, 10000, 100, 1000])
LOWER = np.array([100, 100, 1, 10])
DEFAULT = np.array([5000, 5000, 30, 100])
PARAMS = ['primitive_collision', 'robot_self_collision', 'manipulability', 'stop_cost']

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

print(f'printing results... {type(res)}')
for r in res:
    print(r)