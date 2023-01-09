import math
import numpy as np
from generator import State


class Simulator:
	def __init__(self, h_values, h_probs, lambdas, bits=[4, 6], beta=0.01, weights=[0.5, 0.5]):
		self.h_values = h_values
		self.h_probs = h_probs
		self.lambdas = lambdas
		self.bits = bits
		self.beta = beta
		self.weights = weights
		self.vaoi = [[], []]
		self.wvaoi = []

	def getNextState(self, current_state, action):
		deltas = []
		for i, a in enumerate(action):
			deltas.append(current_state.deltas[i] * (1 - a) + np.random.binomial(1, self.lambdas[i]))

		hs = []
		for i in range(2):
			idx = np.random.binomial(1, self.h_probs[1])
			hs.append(self.h_values[idx])

		return State(deltas, hs)

	def updateVAOI(self, current_state):
		for i in range(2):
			self.vaoi[i].append(self.weights[i]*current_state.deltas[i])
		self.wvaoi.append(self.weights[0]*current_state.deltas[0] + self.weights[1]*current_state.deltas[1])

	def evaluate(self, policy, n=1_00_000):
		current_state = State([0, 0], [self.h_values[0], self.h_values[0]])

		for _ in range(n):
			action = policy.getAction(current_state)
			next_state = self.getNextState(current_state, action)
			self.updateVAOI(current_state)

			current_state = next_state

			