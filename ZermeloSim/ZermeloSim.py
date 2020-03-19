from ZermeloSim.ZermeloAgent import Agent
from ZermeloSim.ZermeloEnvironment import Env
from threading import Thread
from queue import Queue
import multiprocessing


def _run_agent(agent, env):
    # Initialise the success flag for the agent
    success = False
    # Initialise the distance cost
    d = 0
    # Initialise the energy cost
    e = 0
    # Initialise the time cost
    t = 0
    # Loop until success
    while not success:
        # Get the state of the agent within the environment
        inputs = env.get()
        # Get the control velocity
        w = agent.step(inputs.T)
        # Get the updated environment
        success, distance, energy, time = env.update(w.T)
        # Increment the distance cost
        d = d + distance
        # Increment the energy cost
        e = e + energy
        # Increment the time cost
        t = t + time
    return d, e, t


class Simulator:
    def __init__(self, n, num_workers=4):
        # The population size
        self._n = n
        # The number of workers
        if num_workers > n:
            self._num_workers = n
        else:
            self._num_workers = num_workers
        # Initialize the environments
        self.envs = [Env() for _ in range(n)]
        # Initialize the agents
        self.agents = [Agent() for _ in range(n)]
        # Initialize the simulations
        self.simulations = Queue()
        # Initialize the results
        self.results = Queue()

    def _worker(self):
        while True:
            # Get a job
            job = self.simulations.get()
            # End worker if the job is None
            if job is None:
                self.simulations.task_done()
                break
            # Decompose job into id, agent and env objects
            idx, agent, env = job
            print("Simulation {}... processing.".format(idx))
            # Run a simulation on the agent and environment, get costs
            d, e, t = _run_agent(agent, env)
            # Bundle results with idx and place in results queue
            self.results.put((idx, d, e, t))
            print("Simulation {}... done.".format(idx))
            # Flag the task as done for the simulations queue
            self.simulations.task_done()

    def reset(self):
        for env in self.envs:
            env.reset()

    def run_sim(self):
        # Initialise the workers
        threads = []
        for i in range(self._num_workers):
            t = Thread(target=self._worker)
            t.start()
            threads.append(t)
        # Each simulation is a tuple containing: (id, agent, environment)
        for sim in zip(list(range(self._n)), self.agents, self.envs):
            self.simulations.put(sim)
        # Block until all simulations in queue are processed
        self.simulations.join()
        # Close the workers
        for _ in range(self._num_workers):
            self.simulations.put(None)
        # Block until all simulations in queue are processed
        self.simulations.join()
        for t in threads:
            t.join()
        # Results queue should be populated, unpack.
        results = []
        for _ in range(self._n):
            results.append(self.results.get())
        results.sort()
        return results


def worker(jobs, results):
    # Initialise the agent
    agent = Agent()
    # Initialise the environment
    env = Env()
    # Begin the loop
    while True:
        # Wait for a job to come in
        j = jobs.get()
        # If the job is None then finish the process
        if j is None:
            break
        # Unpack the job
        idx, weights = j
        print("Simulation {}... processing.".format(idx))
        # If the weights are a list, then update the agent weights
        if isinstance(weights, list):
            agent.model.set_weights(weights)
        # Run the simulation
        costs = _run_agent(agent, env)
        # Pack the results and send
        weights = agent.model.get_weights()
        results.put((idx, weights) + costs)
        # reset the environment
        env.reset()
        # Notify the job queue that the simulation is done
        print("Simulation {}... done.".format(idx))
        jobs.task_done()
    # Notify the job queue that the exit command is done
    jobs.task_done()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]