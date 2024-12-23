class AgentFactory:
    @staticmethod
    def create_agent(algorithm, state_size, action_size, **kwargs):
        """
        Dynamically create an agent based on the algorithm name.

        :param algorithm: Name of the algorithm (e.g., 'DQN', 'PPO').
        :param state_size: Size of the state space.
        :param action_size: Size of the action space.
        :param kwargs: Additional arguments for specific agents.
        :return: An instantiated agent.
        """
        if algorithm == "DQN":
            from algorithms import DQNAgent
            return DQNAgent(state_size, action_size, **kwargs)
        elif algorithm == "XXX":
            pass
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")