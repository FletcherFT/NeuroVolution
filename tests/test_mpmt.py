from multiprocessing import Process, cpu_count
from multiprocessing import JoinableQueue as JQueue
from threading import Thread
from queue import Queue
from ZermeloSim.ZermeloSim import worker
import time
start = time.time()


def proc(jobs, results, t):
    thread_jobs = Queue()
    threads = [Thread(target=worker(thread_jobs, results)).start() for _ in range(t)]
    while True:
        job = jobs.get()
        if job is None:
            break
        thread_jobs.put(job)
    for _ in range(t):
        thread_jobs.put(None)
    thread_jobs.join()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    n = 100
    p = cpu_count()-1
    t = 4
    jobs = JQueue()
    results = JQueue()
    pool = [Process(target=proc, args=(jobs, results, t)).start() for _ in range(p)]
    idx = range(n)
    weights = [-1]*n
    for job in zip(idx, weights):
        jobs.put(job)
    jobs.join()
    r = [results.get() for _ in range(n)]
    for _ in range(p):
        jobs.put(None)
    jobs.join()
    print("Multi-Processing With Threading Took: {} secs".format(time.time()-start))
