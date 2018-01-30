from yahoo_finance import Share
from matplotlib import pyplot as plt
import numpy as np
import random
import tensorflow as tf
import random


class DecisionPolicy:
    def select_action(self, current_state, step):
        pass

    def update_q(self, state, action, reward, next_state):
        pass


class RandomDecisionPolicy(DecisionPolicy): #Inherit from DecisionPolicy to implement its functions
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, step): #Randomly choose the next action
        action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action


class QLearningDecisionPolicy(DecisionPolicy):
    def __init__(self, actions, input_dim):
        self.epsilon = 0.9
        self.gamma = 0.001
        self.actions = actions
        output_dim = len(actions)
        h1_dim = 200

        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [output_dim])
        W1 = tf.Variable(tf.random_normal([input_dim, h1_dim]))
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim]))
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
        W2 = tf.Variable(tf.random_normal([h1_dim, output_dim]))
        b2 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2)

        loss = tf.square(self.y - self.q)
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def select_action(self, current_state, step):
        threshold = min(self.epsilon, step / 1000.)
        if random.random() < threshold:
            # Exploit best option with probability epsilon
            action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
            action_idx = np.argmax(action_q_vals)  # TODO: replace w/ tensorflow's argmax
            action = self.actions[action_idx]
        else:
            # Explore random option with probability 1 - epsilon
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action

    def update_q(self, state, action, reward, next_state):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state})
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        next_action_idx = np.argmax(next_action_q_vals)
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx]
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals})


def run_simulation(policy, initial_budget, initial_num_stocks, prices, hist, debug=False):
    budget = initial_budget #Initialize values that depend on computing the net worth of a portfolio
    num_stocks = initial_num_stocks #Initialize values that depend on computing the net worth of a portfolio
    share_value = 0 #Initialize values that depend on computing the net worth of a portfolio
    transitions = list()
    for i in range(len(prices) - hist - 1):
        if i % 100 == 0:
            print('progress {:.2f}%'.format(float(100*i) / (len(prices) - hist - 1)))
        current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks))) #The state is a `hist+2` dimensional vector. We’ll force it to by a numpy matrix.
        current_portfolio = budget + num_stocks * share_value #Calculate the portfolio value
        action = policy.select_action(current_state, i) #Select an action from the current policy
        share_value = float(prices[i + hist + 1])
        if action == 'Buy' and budget >= share_value: #Update portfolio values based on action
            budget -= share_value
            num_stocks += 1
        elif action == 'Sell' and num_stocks > 0: #Update portfolio values based on action
            budget += share_value
            num_stocks -= 1
        else: #Update portfolio values based on action
            action = 'Hold'
        new_portfolio = budget + num_stocks * share_value #Compute new portfolio value after taking action
        reward = new_portfolio - current_portfolio #Compute the reward from taking an action at a state
        next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1], budget, num_stocks)))
        transitions.append((current_state, action, reward, next_state))
        policy.update_q(current_state, action, reward, next_state) #Update the policy after experiencing a new action

    portfolio = budget + num_stocks * share_value #Compute final portfolio worth
    if debug:
        print('${}\t{} shares'.format(budget, num_stocks))
    return portfolio


def run_simulations(policy, budget, num_stocks, prices, hist):
    num_tries = 10 #Compute final portfolio worth
    final_portfolios = list() #Store portfolio worth of each run in this array
    for i in range(num_tries):
        final_portfolio = run_simulation(policy, budget, num_stocks, prices, hist) #Run this simulation
        final_portfolios.append(final_portfolio)
    avg, std = np.mean(final_portfolios), np.std(final_portfolios)
    return avg, std


def get_prices(share_symbol, start_date, end_date, cache_filename='stock_prices.npy'):
    try:
        stock_prices = np.load(cache_filename)
    except IOError:
        share = Share(share_symbol)
        stock_hist = share.get_historical(start_date, end_date)
        stock_prices = [stock_price['Open'] for stock_price in stock_hist]
        np.save(cache_filename, stock_prices)

    return stock_prices


def plot_prices(prices):
    plt.title('Opening stock prices')
    plt.xlabel('day')
    plt.ylabel('price ($)')
    plt.plot(prices)
    plt.savefig('prices.png')


if __name__ == '__main__':
    prices = get_prices('MSFT', '1992-07-22', '2016-07-22')
    plot_prices(prices)
    actions = ['Buy', 'Sell', 'Hold'] #Define the list of actions the agent can take
    hist = 200
    # policy = RandomDecisionPolicy(actions)
    policy = QLearningDecisionPolicy(actions, hist + 2)
    budget = 1000.0
    num_stocks = 0
    avg, std = run_simulations(policy, budget, num_stocks, prices, hist)
    print(avg, std)

