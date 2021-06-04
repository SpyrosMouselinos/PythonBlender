import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import trange


class UCBBandit:
    def __init__(self, k_arm=10, epsilon=0.1, initial=0.0, c=1):
        self.k = k_arm
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.epsilon = epsilon
        self.initial = initial
        self.c = c

    def reset(self):
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)
        self.time = 0
        return

    # Choose the best action
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        UCB_estimation = self.q_estimation + \
                         self.c * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
        q_best = np.max(UCB_estimation)
        return np.where(UCB_estimation == q_best)[0][0]

    # Take an action, update estimation for this action
    def update_step(self, action, reward):
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        return


class TSBandit:
    def __init__(self, k_arm=10, epsilon=0.1, initial=0):
        self.k = k_arm
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        self.q_estimation = np.zeros(self.k) + self.initial
        self.wins = np.zeros(self.k)
        self.trials = np.zeros(self.k)
        self.priors = [(1, 1)] * self.k
        self.time = 0
        return

    # Choose the best action
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        TS_estimation = np.array([np.random.beta(a, b) for (a, b) in self.priors])
        q_best = np.max(TS_estimation)
        return np.where(TS_estimation == q_best)[0][0]

    # Take an action, update estimation for this action
    def update_step(self, action, reward):
        self.time += 1
        self.trials[action] += 1
        self.wins[action] += reward
        self.average_reward += (reward - self.average_reward) / self.time
        self.update_posterior(action)
        self.q_estimation[action] = ((self.trials[action] - 1) / float(self.trials[action])) * self.q_estimation[
            action] + (1 / float(self.trials[action])) * reward
        return

    def update_posterior(self, action):
        action_alpha = 1 + self.wins[action]
        action_beta = 1 + self.trials[action] - self.wins[action]
        self.priors[action] = (action_alpha, action_beta)
        return


# The Backend for Contextual Thompson Sampling
class OnlineLogisticRegression:
    def __init__(self, lambda_, alpha, n_dim):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.n_dim = n_dim,
        self.m = np.zeros(self.n_dim)
        self.q = np.ones(self.n_dim) * self.lambda_
        self.w = np.random.normal(self.m, self.alpha * self.q ** (-1.0), size=self.n_dim)

    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum(
            [np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])

    def grad(self, w, *args):
        X, y = args
        return self.q * (w - self.m) + (-1) * np.array(
            [y[j] * X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)

    def get_weights(self):
        return np.random.normal(self.m, self.alpha * self.q ** (-1.0), size=self.n_dim)

    def fit(self, X, y):
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B",
                          options={'maxiter': 20, 'disp': True}).x
        self.m = self.w
        P = (1 + np.exp(-1 * X.dot(self.m))) ** (-1)
        self.q = self.q + (P * (1 - P)).dot(X ** 2)

    def predict_proba(self, X, mode='sample'):
        self.w = self.get_weights()

        # using weight depending on mode
        if mode == 'sample':
            w = self.w  # weights are samples of posteriors
        elif mode == 'expected':
            w = self.m  # weights are expected values of posteriors
        else:
            raise Exception('mode not recognized!')

        # calculating probabilities
        proba = 1 / (1 + np.exp(-1 * X.dot(w)))
        return np.array([1 - proba, proba]).T


# now, we define a class for our policy
class ThompsonSamplingLR:

    # initializing policy parameters
    def __init__(self, k_arm=10, n_dim=1, epsilon=0.1, initial=0, lambda_=1, alpha=1, buffer_size=200):

        # storing the parameters
        self.lambda_ = lambda_
        self.k = k_arm
        self.alpha = alpha
        self.n_dim = n_dim
        self.buffer_size = buffer_size
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.epsilon = epsilon
        self.initial = initial

    def reset(self):
        self.q_estimation = np.zeros(self.k) + self.initial
        self.trials = np.zeros(self.k)
        self.time = 0
        return

    # Function to fit and predict from a DataFrame as Cache
    def fit_predict(self, data, actual_x):
        olr = OnlineLogisticRegression(self.lambda_, self.alpha, self.n_dim)
        olr.fit(data['x'].values.reshape(-1, 1), data['reward'].values)
        out_df = pd.DataFrame({'prob': olr.predict_proba(np.array(actual_x))[0][1],
                               'm': olr.m, 'q': olr.q * self.alpha ** (-1.0)})
        return out_df

    def act(self, round_df, actual_x):

        # enforcing buffer size
        round_df = round_df.tail(self.buffer_size)

        # if we have enough data, calculate best bandit
        if round_df.groupby(['k', 'reward']).size().shape[0] == 4:

            # predictinng for two of our datasets
            self.ts_model_df = (round_df
                                .groupby('k')
                                .apply(self.fit_predict, actual_x=actual_x)
                                .reset_index().drop('level_1', axis=1).set_index('k'))

            # get best bandit
            action = int(self.ts_model_df['prob'].idxmax())

        # if we do not have, the best bandit will be random
        else:
            action = int(np.random.choice(list(range(self.k)), 1)[0])
            self.ts_model_df = pd.DataFrame({'prob': 0.50, 'm': 0.0, 'q': self.lambda_}, index=[0])

        return action


def test_ucb_bandit_single_step(trials=1000, runs=1000):
    # 7 - ARM Machine with Beta distributions #
    # Best Choice: Arm 7 #

    # Set 1 #
    test_arm_params = [(1, 5), (2, 6), (4, 5), (6, 7), (7, 7), (8, 7), (9, 6)]
    best_choice = 6

    # Set 2 #
    test_arm_params_2 = [0.1, 0.2, 0.1, 0.3, 0.4, 0.35, 0.5]
    best_choice = 6

    def get_beta_reward(params) -> float:
        a, b = params
        return np.random.beta(a, b, size=1)

    def get_gauss_reward(params) -> float:
        a = params
        return np.random.randn() + a

    # Initialize Bandits #
    test_bandits = [UCBBandit(k_arm=7, epsilon=0.2, initial=0, c=1),
                    UCBBandit(k_arm=7, epsilon=0.1, initial=0, c=1),
                    UCBBandit(k_arm=7, epsilon=0.1, initial=0, c=2)]

    rewards = np.zeros((trials, 3, runs))
    n_best_choice = np.zeros((trials, 3, 1))
    for trial in trange(trials):
        for bandit in test_bandits:
            bandit.reset()
        for t in range(runs):
            for bi, bandit in enumerate(test_bandits):
                choice = bandit.act()
                if choice == best_choice:
                    n_best_choice[trial, bi] += 1
                reward = get_gauss_reward(test_arm_params_2[choice])
                bandit.update_step(choice, reward)
                rewards[trial, bi, t] = reward

    rewards = rewards.mean(axis=0)
    plt.figure(figsize=(10, 10))
    plt.title("Sum of Rewards")
    plt.plot(rewards[0], 'r', label='Rewards Bandit 1')
    plt.plot(rewards[1], 'b', label='Rewards Bandit 2')
    plt.plot(rewards[2], 'g', label='Rewards Bandit 3')
    plt.legend(loc=1)
    plt.show()
    plt.close()

    print(f"Agent 1: Estimation: {test_bandits[0].q_estimation}")
    print(f"Agent 2: Estimation: {test_bandits[1].q_estimation}")
    print(f"Agent 3: Estimation: {test_bandits[2].q_estimation}")

    return


def test_ts_bandit_single_step(trials=1000, runs=1000):
    # 7 - ARM Machine with Uniform distributions #
    # Best Choice: Arm 7 #

    # Set 1 #
    test_arm_params = [0.1, 0.2, 0.1, 0.3, 0.35, 0.35, 0.5]
    best_choice = 6

    def get_discrete_reward(params) -> float:
        p = params
        return np.random.binomial(1, p)

    # Initialize Bandits #
    test_bandits = [TSBandit(k_arm=7, epsilon=0.0),
                    TSBandit(k_arm=7, epsilon=0.1),
                    TSBandit(k_arm=7, epsilon=0.2)]

    rewards = np.zeros((trials, 3, runs))
    n_best_choice = np.zeros((trials, 3, 1))
    for trial in trange(trials):
        for bandit in test_bandits:
            bandit.reset()
        for t in range(runs):
            for bi, bandit in enumerate(test_bandits):
                choice = bandit.act()
                if choice == best_choice:
                    n_best_choice[trial, bi] += 1
                reward = get_discrete_reward(test_arm_params[choice])
                bandit.update_step(choice, reward)
                rewards[trial, bi, t] = reward

    rewards = rewards.mean(axis=0)
    plt.figure(figsize=(10, 10))
    plt.title("Sum of Rewards")
    plt.plot(rewards[0], 'r', label='Rewards Bandit 1')
    plt.plot(rewards[1], 'b', label='Rewards Bandit 2')
    plt.plot(rewards[2], 'g', label='Rewards Bandit 3')
    plt.legend(loc=1)
    plt.show()
    plt.close()

    print(f"Agent 1: Estimation: {test_bandits[0].q_estimation}")
    print(f"Agent 2: Estimation: {test_bandits[1].q_estimation}")
    print(f"Agent 3: Estimation: {test_bandits[2].q_estimation}")
    return


def test_olr():
    X_1 = np.linspace(-3, +3, 1000)
    X_2 = np.linspace(-3, +3, 1000)
    noise = np.random.normal(0, 0.2, size=1000)
    y = noise + 0 * X_1 + 1.6 * X_2
    y_log = 1 / (1 + np.exp(-y))
    olr = OnlineLogisticRegression(1, 1, 2)




