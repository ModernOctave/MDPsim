from generator import Generator, genStates
from mdptoolbox.mdp import PolicyIteration


class MDPPolicy:
	def __init__(self, states, h_values, h_probs, lambdas, bits=[4, 4], beta=10):
		self.g = Generator(states, h_values, h_probs, lambdas, bits, beta)
		self.pi = PolicyIteration(self.g.transitions, self.g.rewards, 0.9)

	def getAction(self, stateid):
		return self.g.actions[self.pi.policy[stateid]]