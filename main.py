from simulator import Simulator
from policies import MDPPolicy
from generator import genStates
from matplotlib import pyplot as plt

if __name__ == "__main__":
	# betas = [0.1, 1, 10, 100, 1000, 10000]
	# states = genStates(20, [0.5, 1])

	# costs = []
	# for beta in betas:
	# 	MDP = MDPPolicy(states, [0.5, 1], [0.5, 0.5], [0.5, 0.5], beta=beta)
	# 	sim = Simulator(states, [0.5, 1], [0.5, 0.5], [0.5, 0.5], beta=beta)
	# 	cost = sim.evaluate(MDP)
	# 	costs.append(cost)
	# 	print(f"beta: {beta} cost:{cost}")

	# plt.plot(betas, costs)
	# plt.ylim(0, 4)
	# plt.show()

	states = genStates(10, [0.5, 0.5])
	MDP = MDPPolicy(states, [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], beta=1_00_000)
	for i, state in enumerate(states):
		print(f"state: {state} action: {MDP.getAction(i)}")