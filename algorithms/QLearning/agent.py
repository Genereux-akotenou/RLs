# IMPORT UTILS
import gym, random, warnings, os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.8, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_max=1.0, epsilon_decay=0.003):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_size, action_size))  # Q-table

    def generate_random_excluding(self, range_start, range_end, exclude_list):
        valid_numbers = [num for num in range(range_start, range_end) if num not in exclude_list]
        if not valid_numbers:
            raise ValueError("No valid numbers available in the specified range.")
        return random.choice(valid_numbers)

    def act(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        return np.argmax(self.q_table[state, :])  # Exploitation
    
        # exp_exp_tradeoff = random.uniform(0, 1)
        # if exp_exp_tradeoff > self.epsilon:
        #     action = np.argmax(self.q_table[state,:])
        # else:
        #     action = env.action_space.sample()
        # return action

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[state, action])

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
        """
        Choose the best action based solely on the Q-table (pure exploitation).
        No exploration is performed.
        """
        return np.argmax(self.q_table[state, :])



if __name__ == "__main__":
    # Create the environment
    env = gym.make("FrozenLake")
    state_size = env.observation_space.n  # Number of states in the environment
    action_size = env.action_space.n      # Number of actions in the environment

    # Initialize the QLearningAgent
    agent = QLearningAgent(state_size, action_size, learning_rate=0.8, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_max=1.0, epsilon_decay=0.01)

    # Set training parameters
    episodes = 5000       # Number of training episodes
    max_steps = 100       # Maximum steps per episode
    rewards = []          # To store the rewards for plotting

    # Train the agent
    for episode in tqdm(range(episodes), desc="QL-Train "):
        state = env.reset()[0]
        state = agent.generate_random_excluding(0, state_size, [5, 7, 11, 12])
        env.unwrapped.s = state
        total_reward = 0
        done = False

        for step in range(max_steps):
            # Choose an action a in the current world state (s)
            action = agent.act(state)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            next_state, reward, done, _, _ = env.step(action)
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            agent.update_q_table(state, action, reward, next_state)

            total_reward += reward
            state = next_state
            if done:
                break
        
        # Reduce epsilon (because we need less and less exploration)
        agent.epsilon = agent.epsilon_min + (agent.epsilon_max - agent.epsilon_min)*np.exp(-agent.epsilon_decay*episode) 
        rewards.append(total_reward)
    print ("Score over time: " +  str(sum(rewards)/episodes))

    # Display the Q-table
    print("Final Q-Table:")
    print(agent.q_table)

    # Test the agent after training
    env = gym.make("FrozenLake",)# render_mode="human")
    state = env.reset()[0]
    env.render()
    done = False
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