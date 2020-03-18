from ZermeloSim.ZermeloSim import SimulatorMP


if __name__ == "__main__":
    S = SimulatorMP(3)
    S.start()
    results = S.run_sim()