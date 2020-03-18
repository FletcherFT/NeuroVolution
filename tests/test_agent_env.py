from ZermeloSim.ZermeloAgent import Agent
from ZermeloSim.ZermeloEnvironment import Env


def run_agents(agents, envs):
    # The objectives list, corresponding to performance of each agent
    objectives = []
    # Iterate through each agent
    for agent, env in zip(agents, envs):
        # Initialise the success flag for the agent
        success = False
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
            success, energy, time = env.update(w.T)
            # Increment the energy cost
            e = e + energy
            # Increment the time cost
            t = t + time
        # Give the summed energy and time costs to the objective list.
        objectives.append((e, t))
    return objectives


num_agents = 10
envs = [Env() for _ in range(num_agents)]
agents = [Agent() for _ in range(num_agents)]
objectives = run_agents(agents, envs)
