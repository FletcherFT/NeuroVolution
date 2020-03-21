from matplotlib import pyplot as plt
import numpy as np
import itertools
import csv
import datetime
from pathlib import Path


class ResultsManager:
    def __init__(self):
        self._F = None
        self._axes = None
        self._log = None
        self._plots = []

    def update(self, A, names, **kwargs):
        _, numvars = A.shape
        if self._F is None:
            self._F, self._axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
            self._F.subplots_adjust(hspace=0.05, wspace=0.05)
            for ax in self._axes.flat:
                # Hide all ticks and labels
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                # Set up ticks only on one side for the "edge" subplots...
                if ax.is_first_col():
                    ax.yaxis.set_ticks_position('left')
                if ax.is_last_col():
                    ax.yaxis.set_ticks_position('right')
                if ax.is_first_row():
                    ax.xaxis.set_ticks_position('top')
                if ax.is_last_row():
                    ax.xaxis.set_ticks_position('bottom')
            # Plot the data.
            for i, j in zip(*np.triu_indices_from(self._axes, k=1)):
                for x, y in [(i, j), (j, i)]:
                    p = self._axes[x, y].plot(A[:, x], A[:, y], **kwargs)
                    self._plots += p
            # Label the diagonal subplots...
            for i, label in enumerate(names):
                self._axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                                          ha='center', va='center')
            # Turn on the proper x or y axes ticks.
            for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
                self._axes[j, i].xaxis.set_visible(True)
                self._axes[i, j].yaxis.set_visible(True)
            self._F.canvas.draw()
            self._F.canvas.flush_events()
            return
        c = 0
        for i, j in zip(*np.triu_indices_from(self._axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                self._plots[c].set_xdata(A[:, x])
                self._plots[c].set_ydata(A[:, y])
                c += 1
        self._F.canvas.draw()
        self._F.canvas.flush_events()

    def log(self, A, names):
        if self._log is None:
            logdir = Path("./results").resolve()
            logdir.mkdir(parents=True, exist_ok=True)
            numruns = len(list(logdir.glob("run*")))
            fname = logdir.joinpath("run_{:04d}".format(numruns+1))
            fname.mkdir(parents=True, exist_ok=True)
            fname.joinpath("log.csv")
            self._log = str(fname)
            with open(self._log, "a") as f:
                writer = csv.writer(f, delimeter=',')
                writer.writerow(names)
                writer.writerow(A)
            return
        with open(self._log, "a") as f:
            writer = csv.writer(f, delimeter=',')
            writer.writerow(A)

