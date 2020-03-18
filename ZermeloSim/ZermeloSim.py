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


# TODO TEST SIMTHREAD
class SimThread(Thread):
    """Thread to simulate."""
    def __init__(self, jobs, results):
        # Keep thread job queue
        self._jobs = jobs
        # Keep results job queue
        self._results = results
        # Create an agent
        self._agent = Agent()
        # Create an environment
        self._env = Env()
        # Initialise the Thread class
        super().__init__(target=self._thread())

    def _thread(self):
        # When thread is started, begin infinite loop
        while True:
            # Get a job (one simulation to run)
            job = self._jobs.get()
            # End worker if the job is None
            if job is None:
                break
            # Decompose job into id, agent and env objects
            idx, weight = job
            # If the weight is a list, then update the weights. Otherwise let the random initialisation.
            if isinstance(weight, list):
                self._agent.model.set_weights(weight)
            print("Simulation {}... processing.".format(idx))
            # Run a simulation on the agent and environment, get costs
            d, e, t = _run_agent(self._agent, self._env)
            # Reset the environment
            self._env.reset()
            # Bundle results and weight with idx and place in results queue
            self._results.put((idx, weight, d, e, t))
            print("Simulation {}... done.".format(idx))
            # Flag the task as done for the simulations queue
            self._jobs.task_done()
        self._jobs.task_done()


# TODO TEST SIMPROC
class SimProc(multiprocessing.Process):
    """A process that will spin up some threads."""
    def __init__(self, simulations, results, num_threads=4):
        super().__init__(target=self._proc)
        # Store the number of threads to be generated.
        self._num_threads = num_threads
        # Store the given simulations queue object.
        self._simulations = simulations
        # Store the given results queue object.
        self._results = results
        # Initialise the jobs queue object.
        self._jobs = Queue()
        # Initialise the threads to run.
        self._threads = [SimThread(self._jobs, self._results) for _ in range(self._num_threads)]

    def _proc(self):
        # Start the threads attached to this process
        for t in self._threads:
            t.start()
        # Now loop continuously
        while True:
            # Get a batch of simulations
            batch = self._simulations.get()
            # If the simulation is None, then kill the process
            if batch is None:
                break
            # Put each simulation into the thread job queue
            for sim in batch:
                self._jobs.put(sim)
            # Block until all simulations in batch are done
            self._jobs.join()
            # Mark the simulation batch as done.
            self._simulations.task_done()
        # Send None objects to each thread (so they exit)
        for _ in range(self._num_threads):
            self._jobs.put(None)
        # Block until all items in job queue are processed.
        self._jobs.join()
        # Kill the threads
        for t in self._threads:
            t.join()
        # Mark the close task as done.
        self._simulations.task_done()


class SimulatorMP:
    def __init__(self, n, num_procs=1, num_threads=1):
        # The population size
        self._n = n
        # The number of processes
        self._num_procs = num_procs
        # The number of threads per process
        self._num_threads = num_threads
        # Initialize the simulations
        self._simulations = multiprocessing.Queue()
        # Initialize the results
        self._results = multiprocessing.Queue()
        # Initialise the processes
        self._procs = [SimProc(self._simulations, self._results, self._num_threads) for _ in range(self._num_procs)]
        # Start/Stop Flag
        self._is_running = False

    def start(self):
        # Start the processes
        for p in self._procs:
            p.start()
        self._is_running = True

    def run_sim(self, weights=-1):
        # Check if the processes have been started before putting anything in Queue
        if not self._is_running:
            return
        # If weights is not a list, then send a list of -1's to queue (keep initial random weights for each Agent)
        if not isinstance(weights, list):
            weights = [-1]*self._n
        # Divide the simulations up into jobs for each process
        simulations = list(zip(list(range(self._n)), weights))
        simulations = list(chunks(simulations, self._num_procs))
        # Send the simulations to the processes
        for simulation in simulations:
            self._simulations.put(simulation)
        # Block until all simulations in queue are processed
        self._simulations.join()
        # Results queue should be populated, unpack.
        results = []
        for _ in range(self._n):
            results.append(self._results.get())
        results.sort()
        return results

    def stop(self):
        # Signal close to all processes
        for _ in range(self._num_procs):
            self._simulations.put(None)
        # Block until all simulations in queue are processed
        self._simulations.join()
        # Kill all processes
        for p in self._procs:
            p.join()
        # Set the is_running flag to False
        self._is_running = False


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]