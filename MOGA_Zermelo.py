from multiprocessing import Process, cpu_count
from multiprocessing import JoinableQueue as Queue
from ZermeloSim.ZermeloSim import worker
from MOGA.MOGA import NDSAGA


if __name__ == '__main__':
    # Number of generations
    G = 1000
    # Simulation length (iterations)
    I = 1000
    # population size
    n = 100
    # number of simulations to run at once
    p = cpu_count()-1
    # set up jobs queue for simulation processes
    jobs = Queue()
    # set up results queue for simulation processes
    results = Queue()
    # Start the simulation processes
    pool = [Process(target=worker, args=(jobs, results, I)).start() for _ in range(p)]
    # Get the indices
    idx = range(n)
    # Initialise the weights
    weights = [-1]*n
    MOGA = NDSAGA(n, 0.9, 0.02)
    # Begin MOGA Loop Here
    for g in range(G):
        # STEP 1: GET FITNESSES
        print("Iteration: {}".format(g))
        # Place the jobs into the simulation queue
        for job in zip(idx, weights):
            jobs.put(job)
        # Wait until all simulations are done
        r = [results.get() for _ in range(n)]
        # Sort the results
        r.sort()
        # STEP 2: GET SOLUTION UPDATES
        weights = MOGA.update(r)
    # The final generation:
    # Place the jobs into the simulation queue
    for job in zip(idx, weights):
        jobs.put(job)
    # Wait until all simulations are done
    jobs.join()
    r = [results.get() for _ in range(n)]
    # Close the processes
    for _ in range(p):
        jobs.put(None)
    jobs.join()
