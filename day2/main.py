import numpy as np
from solvers import VI, PI, LP
from utils import print_result

"""
MDP components
P      : transition matrix of shape (|S| * |A|, |S|) where P[i * |S| + k, j] = p(s_k | s_j, a_i)
            i.e. P = [ P(a_0)^T | P(a_1)^T | ... ]^T
R      : reward matrix of shape (|S|, |A|) where R[i, j] = r(s_i, a_j)
gamma  : discount factor
"""

R = np.array([[-2.0, -0.5],
              [-1.0, -3.0]])

P = np.array([[0.75, 0.25],
              [0.75, 0.25],
              [0.25, 0.75],
              [0.25, 0.75]])

gamma = 0.9

print('\n', end='')
print('*****Discounted Reward Problem*****')
print('\n', end='')

print('-----Value Iteration-----')
v, pi = VI(P, R, gamma)
print_result(v, pi, mode='discount')
print('\n', end='')

print('-----Policy Iteration-----')
v, pi = PI(P, R, gamma)
print_result(v, pi, mode='discount')
print('\n', end='')

print('-----Linear Programming-----')
v, pi = LP(P, R, gamma)
print_result(v, pi, mode='discount')
print('\n')

