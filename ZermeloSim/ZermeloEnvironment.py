import numpy as np


class Env:
    def __init__(self, dt=0.05, m=0, f=0, max_iters=1000):
        """Initialise a shear flow environment."""
        # The initial starting position
        self._X = np.zeros([2, 1])
        # The goal position
        self._G = np.array([[0], [1]])
        # The time step
        self._dt = dt
        # The control vector
        self._w = np.zeros([2, 1])
        # The current gradient
        self._m = m
        # The current velocity at starting point
        self._f = f
        # Initialise the current at starting position
        self._v = np.array([[self._f], [0]])
        # Initialise the iteration counter
        self._i = 0
        # Store the termination counter
        self._I = max_iters

    def get(self):
        """Get the distance vector to destination, previous speed of the agent, current velocity."""
        return np.vstack((self._G-self._X, self._w, self._v))

    def reset(self):
        """Reset the environment."""
        # The initial starting position
        self._X = np.zeros([2, 1])
        # Initialise the current at starting position
        self._v = np.array([[self._f], [0]])
        # The control vector
        self._w = np.zeros([2, 1])
        # Initialise the iteration counter
        self._i = 0

    def _shearflow(self):
        """Get the velocity components for the current position."""
        # TODO Come up with a more sophisticated current model.
        self._v[0] = self._m*self._X[1] + self._f
        return self._v

    def update(self, w):
        """Given the control vector, update the environment and return the reward."""
        # Calculate the movement
        X_dot = w - self._shearflow()
        # Update the position.
        self._X = self._X + X_dot*self._dt
        # Calculate the distance
        distance = np.linalg.norm(self._G-self._X)
        # Check if the goal has been reached.
        success = distance <= 1e-3
        self._i = self._i + 1
        success = success or self._i == self._I
        # Calculate the energy change due to change in control vector.
        energy = np.linalg.norm(w-self._w)**2
        # Update the control velocity with w
        self._w = w
        return success, distance, energy, self._dt
