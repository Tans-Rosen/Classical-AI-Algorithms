from typing import Tuple
from graph import Graph

import numpy as np
import numpy.typing as npt


class Bandit:

    def __init__(
            self,
            graph: Graph,
            conditional_sigma: float,
            strategy: int,
            value: float,
            N: int
    ):
        self.graph = graph
        self.arms = self.graph.arms
        self.edges = self.graph.edges

        self.conditional_sigma = conditional_sigma
        self.strategy = strategy
        self.value = value
        self.N = N

        self.Qvalues = np.zeros(len(self.arms))
        self.arm_counts = np.zeros(len(self.arms))


    def simulate(
        self
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        regret = np.zeros(self.N)
        best_arm = self.graph.shortest_path_ind()

        for i in range(self.N):
            next_arm_idx = 0
            if self.strategy == 0:
                next_arm_idx = self.choose_arm_egreedy()
            elif self.strategy == 1:
                next_arm_idx = self.choose_arm_edecay(i)
            elif self.strategy == 2:
                next_arm_idx = self.choose_arm_ucb(i + 1)
            else:
                print("not a valid strategy")

            actual_reward = self.pull_arm(next_arm_idx)
            self.arm_counts[next_arm_idx] += 1
            self.Qvalues[next_arm_idx] = self.Qvalues[next_arm_idx] + \
            ((actual_reward - self.Qvalues[next_arm_idx]) / self.arm_counts[next_arm_idx])

            
            if best_arm == next_arm_idx:
                regret[i] = 0
            else:
                best_reward = self.pull_arm(best_arm)
                regret[i] = best_reward - actual_reward
        
        return self.Qvalues, regret


    def choose_arm_egreedy(
        self
    ) -> int:
        if np.random.rand() < self.value:
            return np.random.randint(len(self.Qvalues))
        return np.argmax(self.Qvalues)

    def choose_arm_edecay(
        self,
        t: int
    ) -> int:
        epsilon = min(1, ((self.value * len(self.arms)) / (t + 1)))
        
        if np.random.rand() < epsilon:
            return np.random.randint(len(self.Qvalues))
        return np.argmax(self.Qvalues)


    def choose_arm_ucb(
        self,
        t: int
    ) -> int:
        temp_vals = np.zeros(len(self.Qvalues))
        for i in range(len(self.Qvalues)):
            temp_vals[i] = self.Qvalues[i] + (self.value * ((np.log(t)) / (self.arm_counts[i] + 1))**0.5)
        return np.argmax(temp_vals)


    def pull_arm(
        self,
        idx: int,
    ) -> float:
        reward = 0
        for i in range(len(self.arms[idx]) - 1):
            mu_edge = self.edges[self.arms[idx][i]][self.arms[idx][i + 1]]["mu"]
            conditional_mean = np.log(mu_edge) - 0.5 * (self.conditional_sigma ** 2)
            reward -= np.exp(conditional_mean + self.conditional_sigma * np.random.randn())
        return reward


    def get_path_mean(
        self,
        idx: int,
    ) -> float:
        return -self.graph.all_path_means[idx]