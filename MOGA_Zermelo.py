from multiprocessing import Process, cpu_count
from multiprocessing import JoinableQueue as Queue
from ZermeloSim.ZermeloSim import worker
import numpy as np
import random


def get_top_10(results):
    results = np.array(results)
    sort_idx = np.argsort(results, axis=0)
    dist_idx = sort_idx[:, 1]
    n = len(dist_idx)
    top10 = int(max((float(n) * 0.1), 1))
    return dist_idx[:top10].tolist()


def mutate(agent):
    child_agent = Agent()
    mutation_power = 0.02  # hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    weights = agent.model.get_weights()
    weights = [weight + mutation_power*np.random.randn() for weight in weights]
    child_agent.model.set_weights(weights)
    return child_agent


if __name__ == '__main__':
    n = 100
    p = cpu_count()-1
    jobs = Queue()
    results = Queue()
    pool = [Process(target=worker, args=(jobs, results)).start() for _ in range(p)]
    idx = range(n)
    weights = [-1]*n
    for job in zip(idx, weights):
        jobs.put(job)
    jobs.join()
    r = [results.get() for _ in range(n)]
    for _ in range(p):
        jobs.put(None)
    jobs.join()



if __name__ == "__main__":
    # TODO Set up MOGA.
    # The population size
    N = 10
    # Initialise the simulator
    #S = Simulator(N, num_workers=100)

    S = SimulatorMP(N, num_proc=1)
    # Number of generations
    G = 1
    for i in range(G):
        # Run the simulation, and get the results
        results = S.run_sim()
        # Get the top 10 performers
        top10 = get_top_10(results)
        # Get the bottom 90 performers
        bottom90 = list(set(range(N)) - set(top10))
        # Get the list of agents
        agents = S.agents
        # Modify the remainder
        elites = [agents[t] for t in top10]
        # Replace the members of bottom90 with a mutation of top10
        for idx in bottom90:
            agents[idx] = mutate(random.choice(elites))
        # Update the agents in the simulator
        S.agents = agents
        # Reset the environment
        S.reset()
        idx, distance, energy, time = zip(*results)
        best = top10[0]
        print("{} Best\t{}".format(i, results[best]))
