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
			return self.deltas[0] == other.deltas[0] and self.deltas[1] == other.deltas[1] and self.hs[0] == other.hs[0] and self.hs[1] == other.hs[1]
		return False

class Generator:
	def __init__(self, max_vaoi, h_values, h_probs, lambdas, bits=[4, 6], beta=0.01, weights=[0.5, 0.5]):
		self.max_vaoi = max_vaoi
		self.h_values = h_values
		self.h_probs = h_probs
		self.lambdas = lambdas
		self.bits = bits
		self.beta = beta
		self.weights = weights

		self.genStates()
		self.genActions()
		self.genTransitions()
		self.genRewards()
		self.handleEdgeCases()

	def genActions(self):
		self.actions = []
		for action1 in range(2):
			for action2 in range(2):
				self.actions.append((action1, action2))

	def genTransitions(self):
		self.transitions = np.zeros((len(self.actions), len(self.states), len(self.states)))
		for l, action in enumerate(self.actions):
			for j, from_state in enumerate(self.states):
				for k, to_state in enumerate(self.states):
					# Check validity of transition
					if not (to_state.deltas[0] == from_state.deltas[0] * (1 - action[0]) or to_state.deltas[0] == from_state.deltas[0] * (1 - action[0]) + 1):
						continue
					if not (to_state.deltas[1] == from_state.deltas[1] * (1 - action[1]) or to_state.deltas[1] == from_state.deltas[1] * (1 - action[1]) + 1):
						continue

					# Find arrival probability
					prob = 1
					for i in range(2):
						if to_state.deltas[i] == from_state.deltas[i] * (1 - action[i]):
							prob *= 1-self.lambdas[i]
						elif to_state.deltas[i] == from_state.deltas[i] * (1 - action[i]) + 1:
							prob *= self.lambdas[i]

					# Find h probability
					for i in range(2):
						idx = self.h_values.index(to_state.hs[i])
						prob *= self.h_probs[idx]

					# Add to transition matrix
					self.transitions[l][j][k] = prob

	def genRewards(self):
		def powerUsed(state, us):
			idx1 = np.argmax(state.hs)
			idx2 = np.argmin(state.hs)
			P1 = (math.e**(self.bits[idx1]*us[idx1]) - 1)/state.hs[idx1]
			P2 = (math.e**(self.bits[idx2]*us[idx2]) - 1) * (1 / state.hs[idx2] + P1)
			return self.weights[idx1]*P1 + self.weights[idx2]*P2

		self.rewards = np.zeros((len(self.actions), len(self.states), len(self.states)))
		for i, action in enumerate(self.actions):
			for j, from_state in enumerate(self.states):
				for k, to_state in enumerate(self.states):
					# Add VAOI cost
					self.rewards[i][j][k] -= self.weights[0]*to_state.deltas[0] + self.weights[1]*to_state.deltas[1]
					# Add power cost
					self.rewards[i][j][k] -= self.beta * powerUsed(from_state, action)

	def handleEdgeCases(self):
		# Handle edge cases where no transitions are possible
		for i, action in enumerate(self.actions):
			for j, from_state in enumerate(self.states):
				p = sum(self.transitions[i][j])
				if not p == 1:
					self.transitions[i][j][0] += 1-p
					self.rewards[i][j][0] = -math.inf

	def genStates(self):
		self.states = []
		for delta1 in range(self.max_vaoi + 1):
			for delta2 in range(self.max_vaoi + 1):
				for h1 in self.h_values:
					for h2 in self.h_values:
						self.states.append(State([delta1, delta2], [h1, h2]))