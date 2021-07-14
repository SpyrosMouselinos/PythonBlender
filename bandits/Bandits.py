import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
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

    def batch_act(self, batch_size=1):
        actions = []
        for _ in range(batch_size):
            actions.append(self.act())
        return actions

    # Take an action, update estimation for this action
    def update_step(self, action, reward):
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        return

    # Take Updates in Batch Manner #
    def batch_update_step(self, actions, rewards):
        for action, reward in zip(actions, rewards):
            self.update_step(action, reward)
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

    def batch_act(self, batch_size=1):
        actions = []
        for _ in range(batch_size):
            actions.append(self.act())
        return actions

    def _update_posterior(self, action):
        action_alpha = 1 + self.wins[action]
        action_beta = 1 + self.trials[action] - self.wins[action]
        self.priors[action] = (action_alpha, action_beta)
        return

    # Take an action, update estimation for this action
    def update_step(self, action, reward):
        self.time += 1
        self.trials[action] += 1
        self.wins[action] += reward
        self.average_reward += (reward - self.average_reward) / self.time
        self._update_posterior(action)
        self.q_estimation[action] = ((self.trials[action] - 1) / float(self.trials[action])) * self.q_estimation[
            action] + (1 / float(self.trials[action])) * reward
        return

    # Take Updates in Batch Manner #
    def batch_update_step(self, actions, rewards):
        for action, reward in zip(actions, rewards):
            self.update_step(action, reward)
        return


# The Backend for Contextual Thompson Sampling
class OnlineLogisticRegression:
    def __init__(self, lambda_, alpha, n_dim, intercept=False):
        self.intercept = intercept
        self.lambda_ = lambda_
        self.alpha = alpha
        self.n_dim = n_dim
        self.m = np.zeros(self.n_dim + 1 if intercept else self.n_dim)
        self.q = np.ones((self.n_dim + 1 if intercept else self.n_dim)) * self.lambda_
        self.w = np.random.normal(self.m, self.alpha * self.q ** (-1.0),
                                  size=(self.n_dim + 1 if intercept else self.n_dim))

    @staticmethod
    def expand_dim(var_):
        if np.ndim(var_) == 1:
            var_ = np.expand_dims(var_, 1)
        ones = np.ones(shape=(var_.shape[0], 1))
        exp_var_ = np.concatenate([ones, var_], axis=1)
        return exp_var_

    def loss(self, w, *args):
        X, y = args
        loss_ = 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.mean(
            [np.log(1 + np.exp(-y[j] * w.dot(X[j]) + 1e-7)) for j in range(y.shape[0])])
        return loss_

    def grad(self, w, *args):
        X, y = args
        grad_ = self.q * (w - self.m) + (-1) * np.array(
            [y[j] * X[j] / (1. + np.exp(y[j] * w.dot(X[j]) + 1e-7)) for j in range(y.shape[0])]).sum(axis=0)
        return grad_

    def get_weights(self):
        return np.random.normal(self.m, self.alpha * self.q ** (-1.0),
                                size=(self.n_dim + 1 if self.intercept else self.n_dim))

    def fit(self, X, y):
        if self.intercept:
            X = self.expand_dim(X)
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="CG",
                          options={'maxiter': 1000, 'disp': False}).x
        self.m = self.w
        P = (1 + np.exp(-1 * X.dot(self.m))) ** (-1)
        self.q = self.q + (P * (1 - P)).dot(X ** 2)
        return

    def predict_proba(self, X, mode='sample'):
        if self.intercept:
            X = self.expand_dim(X)
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
        return proba


class TSCBandit:

    # initializing policy parameters
    def __init__(self, k_arm=10, n_dim=1, initial=0, lambda_=1.0, alpha=1.0, intercept=True, buffer_size=1000):

        # storing the parameters
        self.lambda_ = lambda_
        self.k = k_arm
        self.alpha = alpha
        self.n_dim = n_dim
        self.buffer_size = buffer_size
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.initial = initial
        self.intercept = intercept

    def reset(self):
        self.q_estimation = np.zeros(self.k) + self.initial
        self.trials = np.zeros(self.k)
        self.action_count = np.zeros(self.k)
        self.time = 0
        return

    # Function to fit and predict from a DataFrame as Cache
    def fit_predict(self, data, context):
        olr = OnlineLogisticRegression(self.lambda_, self.alpha, self.n_dim, intercept=self.intercept)
        olr.fit(data[[f'X_{f + 1}' for f in range(self.n_dim)]].values, data['reward'].values)
        prob = olr.predict_proba(np.array(context))
        if np.ndim(prob) > 1:
            prob = np.squeeze(prob)
        out_df = pd.DataFrame({'prob': prob})
        return out_df

    def act(self, cache_df, context):

        # enforcing buffer size
        cache_df = cache_df.tail(self.buffer_size)

        # if we have enough data, calculate best bandit
        if cache_df.groupby(['k', 'reward']).size().shape[0] >= self.k * 2:
            self.ts_model_df = (cache_df
                                .groupby('k')
                                .apply(self.fit_predict, context=context)
                                .reset_index().drop('level_1', axis=1).set_index('k'))

            self.ts_model_df['context_id'] = [f for k in [list(np.arange(0, context.shape[0]))] * self.k for f in k]
            action = self.ts_model_df.groupby('context_id')['prob'].idxmax().values
            if action.shape[0] == 1:
                action = int(action[0])

        # if we do not have, the best bandit will be random
        else:
            action = int(np.random.choice(list(range(self.k)), 1)[0])
            self.ts_model_df = pd.DataFrame({'prob': 0.50}, index=[0])

        return action

    def update_step(self, action, reward):
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        return


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
    noise = np.random.normal(0, 0.01, size=1000)

    arm_1_y = 0.01 + 0.5 * X_1 + 0.2 * X_2
    arm_2_y = 0.01 + 0.5 * X_1 + X_2
    arm_1_y = 1 / (1 + np.exp(-arm_1_y))
    arm_2_y = 1 / (1 + np.exp(-arm_2_y))

    X_1 = np.expand_dims(X_1, 1)
    X_2 = np.expand_dims(X_2, 1)
    X = np.concatenate([X_1, X_2], axis=1)
    Xs = [X, X]
    Ys = [arm_1_y, arm_2_y]

    i = 0
    olr = OnlineLogisticRegression(0.5, 1, Xs[i].shape[1], False)
    olr.fit(Xs[i], Ys[i])

    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(0, 1000), Ys[i], 'b', label='Real')
    plt.plot(np.arange(0, 1000), olr.predict_proba(Xs[i], mode='sample'), 'r', label='Predicted (Sample)')
    plt.legend(loc=3)
    plt.show()
    plt.close()


def test_ts_contextual_bandit_single_step(trials=1, runs=200):
    # 5 - ARM Machine with Logistic Context Distributions #
    # Best Choice: Arm 1 #

    X_1 = np.linspace(-3, +3, 1000)
    noise = np.random.normal(0, 0.01, size=1000)

    arm_1_y = noise + 0.3 * X_1
    arm_2_y = noise + 0.25 * X_1
    arm_3_y = noise + 0.2 * X_1
    arm_4_y = noise + 0.15 * X_1
    arm_5_y = noise + 0.1 * X_1

    arm_1_y = 1 / (1 + np.exp(-arm_1_y))
    arm_2_y = 1 / (1 + np.exp(-arm_2_y))
    arm_3_y = 1 / (1 + np.exp(-arm_3_y))
    arm_4_y = 1 / (1 + np.exp(-arm_4_y))
    arm_5_y = 1 / (1 + np.exp(-arm_5_y))

    plt.figure(figsize=(10, 10))
    plt.title("Region of Arm Interest")
    plt.plot(X_1, arm_1_y, label='Arm_1')
    plt.plot(X_1, arm_2_y, label='Arm_2')
    plt.plot(X_1, arm_3_y, label='Arm_3')
    plt.plot(X_1, arm_4_y, label='Arm_4')
    plt.plot(X_1, arm_5_y, label='Arm_5')
    plt.legend()
    plt.show()

    all_results = np.concatenate([np.expand_dims(arm_1_y, 1),
                                  np.expand_dims(arm_2_y, 1),
                                  np.expand_dims(arm_3_y, 1),
                                  np.expand_dims(arm_4_y, 1),
                                  np.expand_dims(arm_5_y, 1)], axis=1)
    winner = np.argmax(all_results, axis=1)
    plt.figure(figsize=(10, 10))
    plt.title("Region of Arm Interest")
    plt.plot(X_1, winner)
    plt.savefig("v.png")
    plt.show()
    plt.close()

    test_arm_params = [(0.0, 0.3), (0.0, 0.25), (0.0, 0.2), (0.0, 0.25), (0.0, 0.15)]
    best_choice = 0

    def get_context():
        x1 = np.random.uniform(-3, 3, size=1)
        return np.array([x1]).reshape(1, -1)

    def get_discrete_reward(context, params) -> float:
        intercept, beta_1 = params
        out = intercept + beta_1 * context[0][0]
        log_out = 1 / (1 + np.exp(-out))
        return np.random.binomial(1, log_out)

    # Initialize Bandits #

    test_bandits = [TSCBandit(k_arm=5, n_dim=1, lambda_=0.5, alpha=1.5, intercept=False),
                    TSBandit(k_arm=5, epsilon=0.1, initial=0)]

    rewards = np.zeros((trials, 2, runs))
    n_best_choice = np.zeros((trials, 2, 1))
    for trial in trange(trials):
        for bandit in test_bandits:
            bandit.reset()
        for bi, bandit in enumerate(test_bandits):
            if bi == 0:
                cache_df = pd.DataFrame({'k': [], 'X_1': [], 'reward': []})
            for t in trange(runs):
                context = get_context()
                if bi == 0:
                    choice = bandit.act(cache_df, context)
                else:
                    choice = bandit.act()
                if choice == best_choice:
                    n_best_choice[trial, bi] += 1
                reward = get_discrete_reward(context=context, params=test_arm_params[choice])
                temp_df = pd.DataFrame({'X_1': context[0][0], 'k': choice, 'reward': reward},
                                       index=[t])

                cache_df = pd.concat([cache_df, temp_df])
                bandit.update_step(choice, reward)
                rewards[trial, bi, t] = bandit.average_reward

    rewards = rewards.mean(axis=0)
    plt.figure(figsize=(10, 10))
    plt.title("Sum of Rewards")
    plt.plot(rewards[0], 'r', label='Rewards Contextual Bandit 1')
    # plt.plot(rewards[1], 'b', label='Rewards Contextual Bandit 2')
    # plt.plot(rewards[2], 'g', label='Rewards Contextual Bandit 3')
    plt.plot(rewards[1], 'c', label='Rewards Contextless Bandit 4')
    plt.legend(loc=1)
    plt.savefig("r.png")
    plt.show()
    plt.close()

    print(f"Agent 1: Estimation: {test_bandits[0].q_estimation}")
    # print(f"Agent 2: Estimation: {test_bandits[1].q_estimation}")
    # print(f"Agent 3: Estimation: {test_bandits[2].q_estimation}")
    print(f"Agent 4: Estimation: {test_bandits[1].q_estimation}")

    plt.figure(figsize=(10, 10))
    inputs = np.arange(-3, 3, 0.1)
    choices = test_bandits[0].act(cache_df, inputs.reshape(-1, 1))
    plt.title("Non-Stationary Choice of Arm")
    plt.plot(inputs, choices, label='Bandit Choice - Bandit 1')
    # plt.plot(inputs, test_bandits[1].olr.predict_proba(inputs), label='Bandit Choice - Bandit 2')
    # plt.plot(inputs, test_bandits[2].olr.predict_proba(inputs), label='Bandit Choice - Bandit 3')
    plt.plot(inputs, [np.argmax(test_bandits[1].q_estimation)] * inputs.shape[0], label='Bandit Choice - Bandit 4')
    plt.savefig("l.png")
    plt.show()
    plt.close()



        
