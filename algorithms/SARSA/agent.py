# IMPORT UTILS
import gym, random, warnings, os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SARSAAgent:
    def __init__(self, state_size, action_size, learning_rate=0.8, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_max=1.0, epsilon_decay=0.003):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.q_table = np.ones((state_size, action_size))  # Q-table
        self.q_table[5,:] = 0.0
        self.q_table[7,:] = 0.0
        self.q_table[11,:] = 0.0
        self.q_table[12,:] = 0.0
        self.q_table[15,:] = 0.0

    def act(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        return np.argmax(self.q_table[state, :])  # Exploitation

    def update_q_table(self, state, action, reward, next_state, next_action):
        """SARSA Q-value update."""
        self.q_table[state, action] += self.learning_rate * (reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action])

    def update_q_table(self, state, action, reward, next_state, next_action):
        """SARSA Q-value update."""
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

    def save_q_table(self, output_path):
        """Save the Q-table to a file."""
        try:
            np.save(output_path, self.q_table)
            print(f"Q-table successfully saved to {output_path}.")
        except Exception as e:
            print(f"An error occurred while saving the Q-table: {e}")

    def load(self, input_path):
        """Load the Q-table from a file."""
        try:
            self.q_table = np.load(input_path)
            print(f"Q-table successfully loaded from {input_path}.")
        except Exception as e:
            print(f"An error occurred while loading the Q-table: {e}")

    def predict(self, state):
        """Choose the best action based solely on the Q-table (pure exploitation)."""
        return np.argmax(self.q_table[state, :])

if __name__ == "__main__":
    # Create the environment
    env = gym.make("FrozenLake")
    state_size = env.observation_space.n  # Number of states in the environment
    action_size = env.action_space.n      # Number of actions in the environment

    # Initialize the SARSAAgent
    agent = SARSAAgent(state_size, action_size, learning_rate=0.8, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_max=1.0, epsilon_decay=0.01)

    # Set training parameters
    episodes = 5000       # Number of training episodes
    max_steps = 100       # Maximum steps per episode
    rewards = []          # To store the rewards for plotting

    # Train the agent using SARSA
    for episode in tqdm(range(episodes), desc="SARSA Training"):
        state = env.reset()[0]
        total_reward = 0
        done = False

        # Choose an action based on the policy
        action = agent.act(state)

        for step in range(max_steps):
            # Take the action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)
            #if next_state in [5, 7, 11, 12]:
            #    reward = -0.1

            # Choose the next action based on the policy
            next_action = agent.act(next_state)

            # Update the Q-value using SARSA update rule
            agent.update_q_table(state, action, reward, next_state, next_action)

            total_reward += reward
            state, action = next_state, next_action

            if done:
                break

        # Reduce epsilon (less exploration as we train more)
        agent.epsilon = agent.epsilon_min + (agent.epsilon_max - agent.epsilon_min) * np.exp(-agent.epsilon_decay * episode)
        rewards.append(total_reward)

    print("Score over time:", sum(rewards) / episodes)

    # Display the Q-table
    print("Final Q-Table:")
    print(agent.q_table)

    # Test the agent after training
    state = env.reset()[0]
    env.render()
    total_reward = 0
    print("\nTesting the agent...\n")
    for _ in range(max_steps):
        action = agent.predict(state)  # Use the trained Q-table to predict the best action
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
        state = next_state
        if done:
            break

    print(f"Total reward during testing: {total_reward}")