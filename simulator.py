import math
import numpy as np
from generator import State


class Simulator:
	def __init__(self, states, h_values, h_probs, lambdas, bits=[4, 6], beta=10):
		self.states = states
		self.h_values = h_values
		self.h_probs = h_probs
		self.lambdas = lambdas
		self.bits = bits
		self.beta = beta

	def getNextState(self, current_state, action):
		deltas = []
		for i, a in enumerate(action):
			deltas.append(self.states[current_state].deltas[i] * (1 - a) + np.random.binomial(1, self.lambdas[i]))

		hs = []
		for i in range(2):
			idx = np.random.binomial(1, self.h_probs[1])
			hs.append(self.h_values[idx])

		next_state = self.states.index(State(deltas, hs))
		return next_state

	def getCost(self, current_state, action):
		def powerUsed(state, us):
			idx1 = np.argmax(state.hs)
			idx2 = np.argmin(state.hs)
			P1 = (math.e**(self.bits[idx1]*us[idx1]) - 1)/state.hs[idx1]
			P2 = (math.e**(self.bits[idx2]*us[idx1]) - 1) * (1 / state.hs[idx2] + P1)
			return P1 + P2

		cost = 0
		for i in range(2):
			cost += 0.5*self.states[current_state].deltas[i]

		# cost += powerUsed(self.states[current_state], action) * self.beta

		return cost

	def evaluate(self, policy, n=1_00_000):
		current_state = 0
		cost = 0

		for _ in range(n):
			action = policy.getAction(current_state)
			next_state = self.getNextState(current_state, action)
			cost += self.getCost(current_state, action)

			current_state = next_state

		return cost / n