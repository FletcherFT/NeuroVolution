from ZermeloSim.ZermeloSim import Simulator
import time
start = time.time()

if __name__ == "__main__":
    # TODO Set up MOGA.
    # The population size
    N = 100
    # Initialise the simulator
    S = Simulator(N, num_workers=7)
    # Test the workers
    results = S.run_sim()
    print("Multi-Threading Took: {} secs".format(time.time()-start))
