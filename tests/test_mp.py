from multiprocessing import Process, cpu_count
from multiprocessing import JoinableQueue as Queue
from ZermeloSim.ZermeloSim import worker
import time
start = time.time()

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
    print("Multi-Processing Took: {} secs".format(time.time()-start))
