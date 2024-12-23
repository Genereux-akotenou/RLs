from algorithms import DQNAgent
import gym
import matplotlib.pyplot as plt
import numpy as np
import os

class FrozenLake():
    def __init__(self, custom_map, is_slippery=False, render_mode="human"):
        self.custom_map = custom_map
        self.env         = gym.make('FrozenLake-v1', desc=self.custom_map, is_slippery=is_slippery)
        self.env_test    = gym.make('FrozenLake-v1', desc=self.custom_map, is_slippery=is_slippery, render_mode=render_mode)
        self.state_size  = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.agent  = DQNAgent(self.state_size, self.action_size)

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
        self.init_model_dir(output_dir)
        reward_list = []

        for episode in range(self.n_episodes):
            state = self.env.reset()[0]
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
                    print(f'Episode: {episode:4}/{self.n_episodes}\t step: {t:4}. Eps: {float(self.agent.epsilon):.2}, reward {reward}')
                    break

            reward_list.append(reward)
            if len(self.agent.memory) > self.batch_size:
                self.agent.train(self.batch_size, episode)
            if episode % 50 == 0:
                self.agent.save(output_dir + f"weights_{episode:04d}.weights.h5")

        print('Train mean % score =', round(100 * np.mean(reward_list), 1))

    def test(self, model_path, test_episodes=10):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.agent.load(model_path)
        test_wins = []
        for episode in range(test_episodes):
            state = self.env_test.reset()[0]
            state = self.encode_state(state)
            done = False
            reward = 0

            print(f"******* EPISODE {episode + 1} *******")
            for step in range(self.max_steps):
                action = self.agent.predict(state)
                new_state, reward, done, info, _ = self.env_test.step(action)
                new_state = self.encode_state(new_state)
                state = new_state

                self.env_test.render()
                if done:
                    print(f"Episode Reward: {reward}")
                    break

            test_wins.append(reward)
        self.env_test.close()
        
        print(f"Test mean % score = {int(100 * np.mean(test_wins))}")
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