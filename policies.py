from generator import Generator
from mdptoolbox.mdp import PolicyIteration
from greedy import Greedy
from SRP import SRP


class MDPPolicy:
	def __init__(self, max_vaoi, h_values, h_probs, lambdas, bits=[4, 4], beta=0.01, weights=[0.5, 0.5]):
		self.g = Generator(max_vaoi, h_values, h_probs, lambdas, bits, beta, weights)
		self.pi = PolicyIteration(self.g.transitions, self.g.rewards, 0.1)

	def getAction(self, state):
		return self.g.actions[self.pi.policy[self.g.states.index(state)]]

class GreedyPolicy:
	def __init__(self, h_values, h_probs, lambdas, bits=[4, 4], beta=0.01, weights=[0.5, 0.5]):
		self.g = Greedy(h_values, h_probs, lambdas, weights, beta, bits)

	def getAction(self, state):
		return self.g.getAction(state)

class SRPPolicy:
	def __init__(self, h_values, h_probs, lambdas, bits=[4, 4], beta=0.01, weights=[0.5, 0.5]):
		self.g = SRP( h_values, h_probs, lambdas, bits, beta, weights)

	def getAction(self, state):
		return self.g.getAction(state)