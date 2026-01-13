"""
A dynamic programming agent for a stochastic task environment
"""

import random
import math
import sys


class DP_Agent(object):

    def __init__(self, states, parameters):
        self.gamma = parameters["gamma"]
        self.V0 = parameters["V0"]

        self.states = states
        self.values = {}
        self.policy = {}

        for state in states:
            self.values[state] = parameters["V0"]
            self.policy[state] = None


    def setEpsilon(self, epsilon):
        pass

    def setDiscount(self, gamma):
        self.gamma = gamma

    def setLearningRate(self, alpha):
        pass


    def choose_action(self, state, valid_actions):
        return self.policy[state]

    def update(self, state, action, reward, successor, valid_actions):
        pass


    def value_iteration(self, valid_actions, transition):
        """ Computes all optimal values using value iteration and stores them in self.values.

        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        keep_going = True
        while keep_going:
            for s in self.states:
                possible_values = []
                for a in valid_actions(s):
                    (successor, reward) = transition(s, a)
                    if successor == None:
                        reward = 0
                    possible_values.append(reward + (self.gamma * self.values[successor]))
                    
                maxval = max(possible_values)
                if abs(self.values[s] - maxval) <= 10e-6:
                    keep_going = False
                    
                self.values[s] = maxval


    def policy_extraction(self, valid_actions, transition):
        """ Computes all optimal actions using value iteration and stores them in self.policy.

        Args:
            valid_actions (Callable): Function that returns a list of actions given a state.
            transition (Callable): Function that returns successor state and reward given a state and action.
        """
        for s in self.states:
            possible_values = []
            for a in valid_actions(s):
                (successor, reward) = transition(s, a)
                possible_values.append((reward + (self.gamma * self.values[successor]), a))
            self.policy[s] = max(possible_values, key=lambda x: x[0])[1]
