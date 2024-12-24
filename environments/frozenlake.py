import gym, os
import matplotlib.pyplot as plt
import numpy as np
from config.agent_factory import AgentFactory
from tqdm import tqdm
import multiprocessing

class FrozenLake():
    def __init__(self, algorithm, custom_map, is_slippery=False, render_mode="human", verbose=0, **agent_kwargs):
        self.custom_map = custom_map
        self.env         = gym.make('FrozenLake-v1', desc=self.custom_map, is_slippery=is_slippery)
        self.env_test    = gym.make('FrozenLake-v1', desc=self.custom_map, is_slippery=is_slippery, render_mode=render_mode)
        self.state_size  = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.agent = AgentFactory.create_agent(algorithm, self.state_size, self.action_size)
        self.verbose=verbose
        self.algorithm=algorithm
        self.hole_position = agent_kwargs.get("hole_position", None)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value <= 0:
            raise ValueError("Batch size must be a positive number.")
        self._batch_size = value

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, value):
        if value <= 0:
            raise ValueError("Max steps must be a positive number.")
        self._max_steps = value

    @property
    def n_episodes(self):
        return self._n_episodes

    @n_episodes.setter
    def n_episodes(self, value):
        if value <= 0:
            raise ValueError("Number of episodes must be a positive number.")
        self._n_episodes = value

    def init_model_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def encode_state(self, state):
        state_encoded = [0] * self.state_size
        state_encoded[state] = 1
        return np.reshape(state_encoded, [1, self.state_size])

    def train(self, output_dir):
        if self.algorithm == "DQN":
            print("Starting training...")
            self.init_model_dir(output_dir)
            reward_list = []

            for episode in tqdm(range(self.n_episodes), desc="Train ") if self.verbose == 0 else range(self.n_episodes):
                state = self.env.reset()[0]
                if self.hole_position != []:
                    if self.hole_position != None:
                        state = self.agent.generate_random_excluding(0, self.state_size, self.hole_position)
                        self.env.unwrapped.s = state
                state = self.encode_state(state)
                reward = 0
                done = False

                for t in range(self.max_steps):
                    action = self.agent.act(state)
                    new_state, reward, done, info, _ = self.env.step(action)
                    new_state = self.encode_state(new_state)
                    self.agent.add_memory(state, action, reward, new_state, done)
                    state = new_state
                    if done:
                        if self.verbose == 1:
                            print(f'Episode: {episode:4}/{self.n_episodes}\t step: {t:4}. Eps: {float(self.agent.epsilon):.2}, reward {reward}')
                        break

                reward_list.append(reward)
                if len(self.agent.memory) > self.batch_size:
                    self.agent.train(self.batch_size, episode)
                if episode % 50 == 0:
                    self.agent.save(output_dir + f"weights_{episode:04d}.weights.h5")

            print('Train mean % score =', round(100 * np.mean(reward_list), 1))
            print(f"Training completed. Model saved to {output_dir}")
        elif self.algorithm == "QLearning":
            self.init_model_dir(output_dir)
            rewards = []
            for episode in tqdm(range(self.n_episodes), desc="QL-Train ") if self.verbose == 0 else range(self.n_episodes):
                state = self.env.reset()[0]
                if self.hole_position != None:
                    state = self.agent.generate_random_excluding(0, self.state_size, self.hole_position)
                    self.env.unwrapped.s = state
                total_reward = 0
                done = False

                for step in range(self.max_steps):
                    # Choose an action a in the current world state (s)
                    action = self.agent.act(state)
                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    next_state, reward, done, _, _ = self.env.step(action)
                    # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                    # qtable[new_state,:] : all the actions we can take from new state
                    self.agent.update_q_table(state, action, reward, next_state)

                    total_reward += reward
                    state = next_state
                    if done:
                        if self.verbose == 1:
                            print(f'Episode: {episode:4}/{self.n_episodes}\t step: {step:4}. Eps: {float(self.agent.epsilon):.2}, reward {total_reward}')
                        break
                
                # Reduce epsilon (because we need less and less exploration)
                self.agent.epsilon = self.agent.epsilon_min + (self.agent.epsilon_max - self.agent.epsilon_min)*np.exp(-self.agent.epsilon_decay*episode) 
                rewards.append(total_reward)

            print ("Score over time: " +  str(sum(rewards)/self.n_episodes))
            self.agent.save_q_table(output_dir + "q_table.npy")

    def test(self, model_path, test_episodes=10):
        print("Starting testing...")
        if self.algorithm == "DQN":
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.agent.load(model_path)
            test_wins = []
            for episode in tqdm(range(test_episodes), desc="Test ") if self.verbose == 0 else range(test_episodes):
                state = self.env_test.reset()[0]
                state = self.encode_state(state)
                done = False
                reward = 0

                if self.verbose == 1:
                    print(f"******* EPISODE {episode + 1} *******")
                for step in range(self.max_steps):
                    action = self.agent.predict(state)
                    new_state, reward, done, info, _ = self.env_test.step(action)
                    new_state = self.encode_state(new_state)
                    state = new_state

                    self.env_test.render()
                    if done:
                        if self.verbose == 1:
                            print(f"Episode Reward: {reward}")
                        break

                test_wins.append(reward)
            self.env_test.close()
            
            print(f"Test mean % score = {int(100 * np.mean(test_wins))}")
            print("Testing completed.")
            self.plot_results(test_wins)

        elif self.algorithm == "QLearning":
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.agent.load(model_path)
            test_wins = []
            for episode in tqdm(range(test_episodes), desc="Test ") if self.verbose == 0 else range(test_episodes):
                state = self.env_test.reset()[0]
                done = False
                reward = 0

                if self.verbose == 1:
                    print(f"******* EPISODE {episode + 1} *******")
                for step in range(self.max_steps):
                    action = self.agent.predict(state)
                    new_state, reward, done, info, _ = self.env_test.step(action)
                    state = new_state

                    self.env_test.render()
                    if done:
                        if self.verbose == 1:
                            print(f"Episode Reward: {reward}")
                        break

                test_wins.append(reward)
            self.env_test.close()
            
            print(f"Test mean % score = {int(100 * np.mean(test_wins))}")
            print("Testing completed.")
            self.plot_results(test_wins)

    def plot_results(self, test_wins):
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(list(range(len(test_wins))), test_wins, s=10)
        plt.title('Frozen Lake Test Results (DQN)')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.ylim((0, 1.1))
        plt.grid()
        plt.savefig('test_results.png', dpi=300)
        plt.show()