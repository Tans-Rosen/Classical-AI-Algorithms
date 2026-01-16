import numpy as np
import numpy.typing as npt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: list = []):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.init = ((1 - self.grid) / np.sum(self.grid)).flatten('F')

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [(i, j), (i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
        neighbors = []
        for a1, a2 in adjacent:
            if 0 <= a1 < M and 0 <= a2 < N and self.grid[a1, a2] == 0:
                neighbors.append((a1, a2))
        return neighbors


    """
    Transition and observation probabilities
    """

    def initT(self):
        """
        Create and return nxn transition matrix, where n = size of grid.
        """
        # TODO


        T_matrix = np.zeros((self.grid.size, self.grid.size)) / self.grid.size
        T_row_num = 0
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                nbrs = self.neighbors((i, j))
                for (k, l) in nbrs:
                    T_matrix[T_row_num][(len(self.grid[i]) * k) + l] = 1 / len(nbrs)
                T_row_num += 1

        return T_matrix

    def initO(self):
        """
        Create and return 16xn matrix of observation probabilities, where n = size of grid.
        """
        # TODO

        O_matrix = np.ones((16, self.grid.size))
        cell_num = 0
        
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                nbrs = self.neighbors((i, j))
                bit1 = '1'
                bit2 = '1'
                bit3 = '1'
                bit4 = '1'
                if (i-1, j) in nbrs:
                    bit1 = '0'
                if (i, j+1) in nbrs:
                    bit2 = '0'
                if (i+1, j) in nbrs:
                    bit3 = '0'
                if (i, j-1) in nbrs:
                    bit4 = '0'
                val = int(bit1 + bit2 + bit3 + bit4, 2)

                for k in range(16):
                    d = (bin(k ^ val)).count('1')
                    prob_e = ((1 - self.epsilon)**(4-d))*(self.epsilon**d)
                    if prob_e == 0.0625:
                        print("prob_e: " + str(prob_e))
                    O_matrix[k][cell_num] = prob_e
                
                cell_num += 1
        return O_matrix


    """
    Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        # TODO

        alpha_p = alpha @ self.trans
        alpha = alpha_p * self.obs[observation, :]
        return alpha


    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current array of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated array.
        """
        # TODO

        beta_p = beta * self.obs[observation, :]
        beta = beta_p @ self.trans.T
        return beta


    def filtering(self, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Alpha vectors at each timestep.
          np.ndarray: Estimated belief state at each timestep.
        """
        # TODO

        alpha_matrix = np.zeros((len(observations), self.grid.size))
        belief_matrix = np.zeros((len(observations), self.grid.size))

        # Initialize with prior distribution and first observation
        alpha_0 = self.init * self.obs[observations[0], :]
        alpha_matrix[0] = alpha_0
        # Normalize
        alpha_sum = np.sum(alpha_matrix[0])
        if alpha_sum > 0:
            alpha_matrix[0] = alpha_matrix[0] / alpha_sum
            belief_matrix[0] = alpha_matrix[0]
        
        for i in range(1, len(observations)):
            alpha_matrix[i] = self.forward(alpha_matrix[i-1], observations[i])
            # Normalize
            alpha_sum = np.sum(alpha_matrix[i])
            if alpha_sum > 0:
                alpha_matrix[i] = alpha_matrix[i] / alpha_sum
                belief_matrix[i] = alpha_matrix[i]

        return alpha_matrix, belief_matrix


    def smoothing(self, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Beta vectors at each timestep.
          np.ndarray: Smoothed belief state at each timestep.
        """
        # TODO

        
        return np.zeros((len(observations), self.grid.size)), np.zeros((len(observations), self.grid.size))


    """
    Parameter learning: Baum-Welch
    """

    def baum_welch(self, observations: list[int]):
        """Learn observation probabilities using the Baum-Welch algorithm.
        Updates self.obs in place.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Learned 16xn matrix of observation probabilities, where n = size of grid.
          list[float]: List of data likelihoods at each iteration.
        """
        # TODO
        self.obs = np.ones((16, self.grid.size)) / 16
        return self.obs, [0]
