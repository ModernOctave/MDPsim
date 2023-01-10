import cvxpy as cp
import numpy as np
import random

class SRP():
  def __init__(self, h_values = [0.5, 1], h_probs = [0.5, 0.5], lambdas=[0.5, 0.5], bits=[4, 6], beta=0.01, weights=[0.5, 0.5]):
    N = 2 # Number of streams
    M = 2 # Number of h values for a given stream

    Lambda = lambdas
    w = weights

    h_i_values = h_values
    h_i_probs = h_probs

    self.h_i_values = h_values = [[x,y] for x in h_i_values for y in h_i_values]
    # print(h_values)
    h_probs = [x*y for x in h_i_probs for y in h_i_probs]
    # print(h_probs)

    # W_values = [[], [1], [2], [1,2]]

    R = bits

    def f_inv(r):
      return np.exp(r) - 1


    P = [[[0,0], [0,0], [0,0], [0,0]], 
                    [[f_inv(R[0])/h_values[0][0], 0], [f_inv(R[0])/h_values[1][0], 0], [f_inv(R[0])/h_values[2][0], 0], [f_inv(R[0])/h_values[3][0], 0]], 
                    [[0,f_inv(R[1])/h_values[0][1]], [0, f_inv(R[1])/h_values[1][1]], [0, f_inv(R[1])/h_values[2][1]], [0, f_inv(R[1])/h_values[3][1]] ],
                    [[f_inv(R[0])/h_values[0][0], f_inv(R[1])/h_values[0][1] + f_inv(R[1])*f_inv(R[0])/h_values[0][0]], [f_inv(R[0])/h_values[1][0] + f_inv(R[0])*f_inv(R[1])/h_values[1][1],f_inv(R[1])/h_values[1][1]], [f_inv(R[0])/h_values[2][0], f_inv(R[1])/h_values[2][1] + f_inv(R[1])*f_inv(R[0])/h_values[2][0] ], [f_inv(R[0])/h_values[3][0], f_inv(R[0])*f_inv(R[1])/h_values[3][1]]]
                  ]


    mu = cp.Variable((2**N,M**N))

    p1 = (P[1][0][0] * (mu[1][0])* h_probs[0] + P[3][0][0] * mu[3][0] * h_probs[0]) + (P[1][1][0] * (mu[1][1])* h_probs[1] + P[3][1][0] * mu[3][1] * h_probs[1]) + (P[1][2][0] * (mu[1][2])* h_probs[2] + P[3][2][0] * mu[3][2] * h_probs[2]) + (P[1][3][0] * (mu[1][3])* h_probs[3] + P[3][3][0] * mu[3][3] * h_probs[3])
    p2 = (P[2][0][1] * (mu[2][0])* h_probs[0] + P[3][0][1] * mu[3][0] * h_probs[0]) + (P[2][1][1] * (mu[2][1])* h_probs[1] + P[3][1][1] * mu[3][1] * h_probs[1]) + (P[2][2][1] * (mu[2][2])* h_probs[2] + P[3][2][1] * mu[3][2] * h_probs[2]) + (P[2][3][1] * (mu[2][3])* h_probs[3] + P[3][3][1] * mu[3][3] * h_probs[3])

    objective = cp.Minimize(w[0] * (Lambda[0] * (cp.inv_pos((mu[1,0] + mu[3,0])* h_probs[0] + (mu[1,1] + mu[3,1])*h_probs[1] + (mu[1,2] + mu[3,2])*h_probs[2] + (mu[1,3] + mu[3,3])*h_probs[3])-1) + beta * p1 ) + w[1] * Lambda[1] * (cp.inv_pos((mu[2,0] + mu[3,0])* h_probs[0] + (mu[2,1] + mu[3,1])*h_probs[1] + (mu[2,2] + mu[3,2])*h_probs[2] + (mu[2,3] + mu[3,3])*h_probs[3])-1) + beta * p2 )
    constraints = [0<=mu, mu<=1, cp.sum(mu[0,:]) == 1, cp.sum(mu[1,:]) == 1, cp.sum(mu[2,:]) == 1, cp.sum(mu[3,:]) == 1]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    
    self.mu = mu.value
    
    
  def getAction(self, State):
    h_vals = State.hs

    h_idx = self.h_i_values.index(h_vals)

    probs = self.mu[:,h_idx]

    action = random.choices([(0,0), (1,0), (0,1), (1,1)], weights=probs)[0]

    return action
    

# print(mu.value)
