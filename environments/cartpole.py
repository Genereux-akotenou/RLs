import gym
import os
import matplotlib.pyplot as plt
import numpy as np
from config.agent_factory import AgentFactory
from tqdm import tqdm

class CartPole:
    def __init__(self, algorithm, render_mode="human", verbose=0, **agent_kwargs):
        self.env = gym.make('CartPole-v1', )#render_mode=render_mode)
        self.env_test = gym.make('CartPole-v1', render_mode=render_mode)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = AgentFactory.create_agent(algorithm, self.state_size, self.action_size, **agent_kwargs)
        self.verbose=verbose
        self.algorithm=algorithm

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
        """For continuous states, normalization or direct use is common."""
        return np.reshape(state, [1, self.state_size])

    def train(self, output_dir):
        if self.algorithm == "DQN":
            print("Starting training...")
            self.init_model_dir(output_dir)
            reward_list = []

            for episode in tqdm(range(self.n_episodes), desc="Train ") if self.verbose == 0 else range(self.n_episodes):
                state = self.env.reset()[0]
                state = self.encode_state(state)
                done = False

                for t in range(self.max_steps):
                    action = self.agent.act(state)
                    new_state, reward, done, _, _ = self.env.step(action)
                    new_state = self.encode_state(new_state)
                    
                    #reward = -1 if done else 1
                    self.agent.add_memory(state, action, reward, new_state, done)
                    state = new_state
                    if done:
                        if self.verbose == 1:
                            print(f'Episode: {episode:4}/{self.n_episodes}\t step: {t:4}. Eps: {float(self.agent.epsilon):.2}')
                        break             

                reward_list.append(t)
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
                # if self.hole_position != None:
                #     state = self.agent.generate_random_excluding(0, self.state_size, self.hole_position)
                #     self.env.unwrapped.s = state
                total_reward = 0
                done = False

                for step in range(self.max_steps):
                    # Choose an action a in the current world state (s)
                    action = self.agent.act(state)
                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    next_state, reward, done, _, _ = self.env.step(action)
                    if done:
                        reward = -1
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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.agent.load(model_path)
        test_rewards = []

        for episode in tqdm(range(test_episodes), desc="Test ") if self.verbose == 0 else range(test_episodes):
            state = self.env_test.reset()[0]
            state = self.encode_state(state)
            total_reward = 0
            done = False

            if self.verbose == 1:
                print(f"******* EPISODE {episode + 1} *******")
            for step in range(self.max_steps):
                action = self.agent.predict(state)
                new_state, reward, done, _, _ = self.env_test.step(action)
                new_state = self.encode_state(new_state)
                state = new_state
                total_reward += reward

                self.env_test.render()
                if done:
                    if self.verbose == 1:
                        print(f"Episode Reward: {total_reward}")
                    break

            test_rewards.append(total_reward)
        self.env_test.close()

        print(f"Test mean % score = {int(100 * np.mean(test_rewards))}")
        print("Testing completed.")
        self.plot_results(test_rewards)

    def plot_results(self, test_rewards):
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(list(range(len(test_rewards))), test_rewards, s=10)
        plt.title('CartPole Test Results')
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.ylim((0, max(test_rewards) + 10))
        plt.grid()
        plt.savefig('test_results.png', dpi=300)
        plt.show()