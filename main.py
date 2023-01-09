from simulator import Simulator
from policies import MDPPolicy
from generator import genStates
from matplotlib import pyplot as plt

if __name__ == "__main__":
	betas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
	betas = [0.005]
	states = genStates(15, [0.5, 1])

	costs = []
	for beta in betas:
		MDP = MDPPolicy(states, [0.5, 1], [0.5, 0.5], [0.5, 0.5], beta=beta)
		sim = Simulator(states, [0.5, 1], [0.5, 0.5], [0.5, 0.5], beta=beta)
		cost = sim.evaluate(MDP)
		costs.append(cost)
		print(f"beta: {beta} cost:{cost}")

	plt.plot(betas, costs)
	plt.ylim(0, 4)
	plt.show()

	# states = genStates(10, [0.5, 0.5])
	# MDP = MDPPolicy(states, [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], beta=1_00_000)
	# for i, state in enumerate(states):
	# 	print(f"state: {state} action: {MDP.getAction(i)}")