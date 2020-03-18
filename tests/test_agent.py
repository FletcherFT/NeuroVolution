from ZermeloSim.ZermeloAgent import Agent
import numpy as np


a = Agent()
b = Agent()

inputs = np.random.uniform(size=(1, 6))
print(a.step(inputs))
print(b.step(inputs))
