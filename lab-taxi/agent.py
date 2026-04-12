import numpy as np                      # Numerical operations (arrays, math)
from collections import defaultdict    # Dictionary with automatic default values

class Agent:

    def __init__(self, nA=6, alpha=0.05, gamma=1.0, epsilon=0.0000001):
        """
        Initialize the agent.

        Params
        ======
        - nA: number of possible actions
        - alpha: learning rate
        - gamma: discount factor
        - epsilon: exploration rate
        """

        self.nA = nA                                   # Total number of actions
        self.Q = defaultdict(lambda: np.zeros(self.nA))# Q-table: maps state -> array of action values
        self.alpha = alpha                             # Learning rate
        self.gamma = gamma                             # Discount factor (future reward importance)
        self.epsilon = epsilon                         # Exploration probability

    def epsilon_greedy_probs(self, Q_s, epsilon):
        """
        Create probability distribution over actions using epsilon-greedy.

        Params
        ======
        - Q_s: Q-values for current state (array of size nA)
        - epsilon: exploration rate

        Returns
        =======
        - probs: probability for each action
        """

        probs = np.ones(self.nA) * (epsilon / self.nA)   # Start with equal probability for all actions
        best_action = np.argmax(Q_s)                    # Find index of best action (highest Q-value)
        probs[best_action] += (1.0 - epsilon)           # Increase probability of best action
        return probs                                   # Return probability distribution

    def select_action(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Params
        ======
        - state: current state

        Returns
        =======
        - action: selected action index
        """

        probs = self.epsilon_greedy_probs(self.Q[state], self.epsilon)  # Get action probabilities
        return np.random.choice(np.arange(self.nA), p=probs)            # Sample action using probs

    def step(self, state, action, reward, next_state, done):
        """
        Update Q-table using SARSA update rule.

        Params
        ======
        - state: previous state
        - action: action taken in that state
        - reward: reward received
        - next_state: resulting state
        - done: whether episode ended
        """

        # If episode is finished, no future reward is considered
        if done:
            target = reward                         # Target is just immediate reward

        else:
            ## Sarsa Approach
            # Select next action using SAME policy (this is SARSA key idea)
            #next_action = self.select_action(next_state)

            # Compute SARSA target: reward + discounted future Q-value
           # target = reward + self.gamma * self.Q[next_state][next_action]
        ## Expected Sarsa Approach
        # Get probability of each action under epsilon-greedy policy best score 9.59, best approach
            policy = self.epsilon_greedy_probs(self.Q[next_state], self.epsilon)

            # Compute expected value over all actions
            expected_q = np.dot(self.Q[next_state], policy)

            # Expected SARSA target
            target = reward + self.gamma * expected_q
       # SARASA MAX
          # max_next_action = np.argmax(self.Q[next_state])
          # target = reward + self.gamma * self.Q[next_state][max_next_action]
        # Update rule (temporal difference learning)
        # Move current Q-value slightly toward target
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
