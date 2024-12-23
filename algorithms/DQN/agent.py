# IMPORT UTILS
import os, random, warnings, gym, logging
import numpy as np
from tqdm import tqdm
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size= state_size
        self.action_size= action_size
        self.memory = deque(maxlen=2500)
        self.gamma=0.9
        self.epsilon=1
        self.epsilon_decay = 0.0003
        self.epsilon_min=0.01
        self.epsilon_max=1
        self.learning_rate=0.001
        self.epsilon_lst=[]
        self.model = self._build_model()

    def _build_model(self):
        """
        Input: State of size state_size
    	•	Hidden Layer: 32 neurons with ReLU activation
    	•	Output: Q-values for each action
    	•	Loss: Mean squared error (MSE)
    	•	Optimizer: Nadam (variant of Adam with Nesterov momentum)
        """
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
        
    def add_memory(self, state, action, reward, next_state, done):
        """
        •	Adds the experience to the memory queue.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        •	Explore: Selects a random action based on epsilon
    	•	Exploit: Predicts Q-values using the model and chooses the action with the highest Q-value
        """
        if np.random.rand() <= self.epsilon:
            #logger.debug("exploration")
            return random.randrange(self.action_size)
        #logger.debug("exploitation")
        return np.argmax(self.model.predict(state, verbose=0))

    def predict(self, state):
        """
        •	Exploit model to prediction next action
        """
        return np.argmax(self.model.predict(state, verbose=0))

    def train(self, batch_size, episode=0):
        """
        •	Samples a minibatch of experiences
    	•	Updates the target Q-value:
            Q(s, a) = r + \gamma \max_{a{\prime}} Q(s{\prime}, a{\prime})
    	•	Trains the model on the updated Q-values
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_state, verbose=0))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            #self.epsilon *= self.epsilon_decay
            #self.epsilon *= self.epsilon_decay
            self.epsilon=(self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay*episode) + self.epsilon_min
        self.epsilon_lst.append(self.epsilon)

    def save(self, name):
        self.model.save_weights(name)
        
    def load(self, name):
        self.model.load_weights(name)