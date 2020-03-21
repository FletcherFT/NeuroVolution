from utils.ResultLogger import ResultsManager
import numpy as np
import time


A = np.random.random(size=(8, 3))
names = ["one", "two", "three"]
R = ResultsManager()
R.update(A, names, color="g", marker=".", markersize=10.0, linestyle='')
for _ in range(10):
    time.sleep(0.5)
    A = np.random.random(size=(8, 3))
    R.update(A, names)