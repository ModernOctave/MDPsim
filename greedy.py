import math
import numpy as np


class State:
	def __init__(self, deltas, hs):
		self.deltas = deltas
		self.hs = hs

	def __str__(self):
		return "(deltas: " + str(self.deltas) + ", hs: " + str(self.hs) + ")"

	def __repr__(self):
		return self.__str__()

	def __eq__(self, other):
		if isinstance(other, State):
			return self.deltas[1] == other.deltas[1] and self.deltas[2] == other.deltas[2] and self.hs[1] == other.hs[1] and self.hs[2] == other.hs[2]
		return False

class Greedy:
	def __init__(self, h_vals, h_probs, arrival_probs, w, beta, bits) -> None:
		self.curr_state = State([0,0], [0.5, 0.5])
		self.actions = [(0,0), (0,1), (1,0), (1,1)]
		self.h_vals = h_vals
		self.h_probs = h_probs
		self.arrival_probs = arrival_probs
		self.w = w
		self.beta = beta
		self.bits = bits
		

	def calculate_cost(self, state, action):
		next_states = []
		next_state_probs = []
		
		for delta_1 in [0,1]:
			for delta_2 in [0,1]:
				for h_1 in [0,1]:
					for h_2 in [0,1]:
						
						next_state = State([state.deltas[0] * (1-action[0]) + delta_1, state.deltas[1] * (1-action[1]) + delta_2], [self.h_vals[h_1], self.h_vals[h_2]])
						next_states.append(next_state)
						
						prob = 1
						if delta_1:
							prob *= self.arrival_probs[0]
						else:
							prob *= 1 - self.arrival_probs[0]
						if delta_2:
							prob *= self.arrival_probs[1]
						else:
							prob *= 1 - self.arrival_probs[1]
						if h_1:
							prob *= self.h_probs[1]
						else:
							prob *= self.h_probs[0]
						if h_2:
							prob *= self.h_probs[1]
						else:
							prob *= self.h_probs[0]

						next_state_probs.append(prob)
						
		cost = 0
		
		for ns, ns_prob in zip(next_states, next_state_probs):
			for i in range(2):
				cost += ns_prob * (self.w[i]*ns.deltas[i])

		cost += self.beta * self.powerUsed(state, action)

		return cost
	
	
	def powerUsed(self, state, us):
			idx1 = np.argmax(state.hs)
			idx2 = np.argmin(state.hs)
			P1 = (math.e**(self.bits[idx1]*us[idx1]) - 1)/state.hs[idx1]
			P2 = (math.e**(self.bits[idx2]*us[idx2]) - 1) * (1 / state.hs[idx2] + P1)
			return self.w[idx1]*P1 + self.w[idx2]*P2
		
		
	def getAction(self, state):
		action_costs = []
		for action in self.actions:
			action_costs.append(self.calculate_cost(state, action))
			
		return self.actions[np.argmin(action_costs)]
	