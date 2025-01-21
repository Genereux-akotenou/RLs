# IMPORT UTILS
import gym, os, random, warnings
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CREATE ENVIROMENT
class EscapeGameEnv(gym.Env):
    def __init__(self):
        super(EscapeGameEnv, self).__init__()
        labels = ["piece1", "piece2", "piece3", "piece4", "piece5", "piece6", "piece7"]
        self.action_space = spaces.Discrete(len(labels))
        self.index_to_label = {i: label for i, label in enumerate(labels)}
        self.observation_space = spaces.Discrete(7)
        self.action_space_dict = {
            # action: ([valid inital state], target_state)
            "piece1": ([2   ], 1),
            "piece2": ([1, 3], 2),
            "piece3": ([2, 6], 3),
            "piece4": ([5   ], 4),
            "piece5": ([4, 6], 5),
            "piece6": ([3, 5], 6),
            "piece7": ([5, 6], 7),
        }
        self.state = None

    def reset(self):
        self.state = 2
        return self.state

    def step(self, action):
        if action not in self.observation_space:
            raise ValueError("Invalid action.")
        valid_state, next_state = self.action_space_dict[self.index_to_label[action]]
        if self.state in valid_state:
            self.state = next_state
            reward = 100 if next_state == 7 else 0
        else:
            reward = -1
        done = self.state == 7
        return self.state-1, reward, done, {'action': self.index_to_label[action], 'new_state': f"Chambre {self.state}"}

    def render(self, mode="human"):
        # Create a simple 2D layout of the chambers
        chamber_positions = {
            1: (1, 3),
            2: (2, 3),
            3: (3, 3),
            4: (1, 2),
            5: (2, 2),
            6: (3, 2),
            7: (2, 1),
        }
        plt.figure(figsize=(6, 6))
        for state, pos in chamber_positions.items():
            color = "red" if state == self.state else "blue"
            plt.scatter(*pos, color=color, s=200)
            plt.text(pos[0], pos[1] + 0.2, f"Chambre {state}", ha="center", fontsize=12)

        plt.xlim(0, 4)
        plt.ylim(0, 4)
        plt.grid(visible=True)
        plt.title("Escape Game Environment")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()

# Q-LERNING SOLVER
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

# Q-LEARNING SOLVER
if __name__ == "__main__":
    # Create the environment
    env = EscapeGameEnv()
    state_size = env.observation_space.n
    action_size = env.action_space.n
    print(state_size)
    print(action_size)

    # Initialize the QLearningAgent
    agent = QLearningAgent(state_size, action_size, learning_rate=0.8, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_max=1.0, epsilon_decay=0.01)

    # Set training parameters
    episodes = 20         # Number of training episodes
    max_steps = 100       # Maximum steps per episode
    rewards = []          # To store the rewards for plotting

    # Train the agent
    for episode in tqdm(range(episodes), desc="QL-Train "):
        state = env.reset()
        #state = agent.generate_random_excluding(0, state_size, [5, 7, 11, 12])
        total_reward = 0
        done = False

        for step in range(max_steps):
            # Choose an action a in the current world state (s)
            action = agent.act(state)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            next_state, reward, done, info = env.step(action)
            print(f"Info: {info}, Done: {done}, , Reward: {reward}")
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
    env = EscapeGameEnv()
    state = env.reset()
    env.render()
    done = False
    total_reward = 0
    max_steps = 50
    for _ in range(max_steps):
        action = agent.predict(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        state = next_state
        if done:
            break
    print(f"Total reward during testing: {total_reward}")