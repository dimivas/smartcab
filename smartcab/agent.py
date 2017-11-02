import argparse
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha=0.3, gamma=0.2, epsilon=0.05):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.reward = None
        self.action = None
        self.q_values = {}
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.sum_reward = 0
        self.num_neg_rewards = 0
        self.num_steps = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.sum_reward = 0
        self.num_neg_rewards = 0
        self.num_steps = 0

    def __initialize_q_values(self, state):
        if state not in self.q_values:
            self.q_values[state] = {}
            for valid_action in self.env.valid_actions:
                self.q_values[state][valid_action] = 0

    def __update_q_values(self, next_state):
        self.__initialize_q_values(next_state)
        if (self.state is not None):
            self.__initialize_q_values(self.state)
            #Q(s, a) <-- Q(s, a) + alpha [r + gamma * max Q(s', a') - Q(s, a)]
            old_value = self.q_values[self.state][self.action]
            learned_value = self.reward + self.gamma * max(self.q_values[next_state].values())
            self.q_values[self.state][self.action] = old_value + self.alpha * (learned_value - old_value)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        current_state = (self.next_waypoint, tuple(inputs.values()))
        self.__update_q_values(current_state)
        self.state = current_state
        
        if (random.random() > self.epsilon):
            action = max(self.q_values[self.state], key=self.q_values[self.state].get)
        else:
            action = random.choice(self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        self.reward = reward
        self.action = action
        self.sum_reward += reward
        self.num_steps += 1
        if (reward < 0.0):
            self.num_neg_rewards += 1
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, sum_rew = {}, neg_rew = {}, steps = {}, next_wp = {}".format(\
            deadline, inputs, action, reward, self.sum_reward, self.num_neg_rewards, self.num_steps, self.next_waypoint)  # [debug]


def run(alpha, gamma, epsilon):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, alpha=alpha, gamma=gamma, epsilon=epsilon)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)

    sim.run(n_trials=100)  # run for a specified number of trials


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("{} not in range [0.0, 1.0]".format(x))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine Learning - Reinforcement Project - SmartCab')
    parser.add_argument('--alpha', type=restricted_float, nargs='?', default=0.3,
                        help='alpha parameter for reinforcement learning algorithm [0.0, 1.0]')
    parser.add_argument('--gamma', type=float, nargs='?', default=0.2,
                        help='gamma parameter for reinforcement learning algorithm [0.0, 1.0]')
    parser.add_argument('--epsilon', type=float, nargs='?', default=0.05,
                        help='epsilon parameter for reinforcement learning algorithm [0.0, 1.0]')
    args = parser.parse_args()

    run(args.alpha, args.gamma, args.epsilon)
