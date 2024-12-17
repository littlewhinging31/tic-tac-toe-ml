import pickle
import random

import numpy as np
from numpy.typing import NDArray

from board import Board
from models.base_model import BaseModel


class ReinforcementLearningModel(BaseModel):
    """
    Reinforcement Learning model implementation.
    """

    def __init__(self, board: Board, alpha: float = 0.1, gamma: float = 0.9):
        """
        Initialize model.

        Parameters:
        board (Board): board.Board instance.
        alpha (float): Hyperparameter, used to soft update the model.
        gamma (float): Discount factor.
        """
        super().__init__(board)

        self.policy = {}

        self.alpha = alpha
        self.gamma = gamma

    def _get_Q(self, state: NDArray[np.int64], action: NDArray[np.int64]):
        """
        Get Q-function value using pre-calculated policy. Returns 0 by default.

        Parameters:
        state (NDArray): Current state of the Board.
        action (NDArray): Action pair.

        Returns:
        float: The value of Q-funciton for given state and action.

        """
        return self.policy.get((tuple(map(tuple, state)), tuple(action)), 0)

    def _compute_Q_max(self, value: int):
        """
        Computes the best possible future value of Q-function. Since we need to take in account
        opponent's response before start calculations, in current implementation we assume that
        opponent plays optimally (concept is borrowed from Minimax approach).

        Parameters:
        value (int): Current player value.

        Returns:
        float: The best possible future value of Q-function.

        """
        available_actions = self.board.calculate_available_actions()

        if not np.any(available_actions):
            return 0

        Q_values = []

        for action in available_actions:
            # opponent make a move
            self.board.make_move(action, -value)

            new_available_actions = self.board.calculate_available_actions()
            Q_values.append(max([self._get_Q(self.board.state, a) for a in new_available_actions], default=0))

            # revert action
            self.board.revert_action(action)

        # opponent plays optimally
        return min(Q_values)

    def _update_policy(self, Q_sa: float, state: NDArray[np.int64], action: NDArray[np.int64], value: int):
        """
        Updates policy with newly calculated Q-function for current state and action.

        Parameters:
        Q_sa (float): Value of Q-function.
        state (NDArray): Current state of the Board.
        action (NDArray): Action pair.
        value (int): Current player value.

        Returns:

        """
        R_s = self._compute_reward(value)

        Q_sa_max = self._compute_Q_max(value)

        Q_new = (1 - self.alpha) * Q_sa + self.alpha * (R_s + self.gamma * Q_sa_max)
        self.policy[(tuple(map(tuple, state)), tuple(action))] = Q_new

    def _make_random_move_and_update_policy(self, value: int):
        """
        Play random move on Board and update policy with newly calculated value.

        Parameters:
        value (int): Current player value.

        Returns:

        """
        random_action = self.board.get_random_action()

        Q_sa = self._get_Q(self.board.state, random_action)

        prev_state = self.board.state.copy()
        self.board.make_move(random_action, value)

        self._update_policy(Q_sa, prev_state, random_action, value)

    def _make_best_move_and_update_policy(self, value: int):
        """
        Play best move on Board and update policy with newly calculated value. Best move
        is calculated using saved policy.

        Parameters:
        value (int): Current player value.

        Returns:

        """
        best_action = self.board.get_random_action()

        Q_sa_best = self._get_Q(self.board.state, best_action)

        if len(self.board.available_actions) == 1:
            self._update_policy(Q_sa_best, self.board.state, best_action, value)
            return

        # find action with larger Q
        for action in self.board.available_actions:
            Q_sa = self._get_Q(self.board.state, action)

            if Q_sa > Q_sa_best:
                best_action = action
                Q_sa_best = Q_sa

        prev_state = self.board.state.copy()
        self.board.make_move(best_action, value)

        self._update_policy(Q_sa_best, prev_state, best_action, value)

    def _compute_reward(self, value: int):
        """
        Compute player's reward for current state.

        Parameters:
        value (int): Current player value.

        Returns:
        (int): Actual value of the reward. It returns 1 if player has won the game, -1 if player
        loose the game on the next move because the opponent hasn't been blocked, 0 otherwise.
        """
        if self.board.is_winner(value):
            return 1

        if self.board.has_winning_move(-value):
            return -1

        return 0

    def load_policy(self):
        """Loads saved policy."""
        with open('models/reinforcement_learning/model.pkl', 'rb') as file:
            self.policy = pickle.load(file)

    def train(self, epsilon: float = 1.0, e_decay: float = 0.85, e_min: float = 0.1, number_of_games: int = 200000):
        """
        Trains a model. Implements epsion-greedy algorithm.

        epsilon (float): Probability of taking a random action by current player.
        e_decay (float): Rate of decreasing epsilon during learning process. Takes more random actions in the beginning
        and less random actions closer to the end.
        e_min (float): Minimal acceptable epsilon value.
        number_of_games (int): Number of games to play.

        Returns:

        """
        for i in range(number_of_games):
            self.board = Board()

            value = self.board.ValueEnum.X.value

            while True:
                random_value = random.random()
                if random_value < epsilon:
                    self._make_random_move_and_update_policy(value)
                else:
                    self._make_best_move_and_update_policy(value)

                if self.board.is_winner(value) or self.board.is_game_over():
                    break

                # switch players
                value = -value

            if (i + 1) % 10000 == 0:
                if (i + 1) > 0:
                    print(f"{i + 1} games played, EPSILON = {epsilon}")

                epsilon *= e_decay
                epsilon = max(e_min, epsilon)

        with open('models/reinforcement_learning/model.pkl', 'wb') as file:
            pickle.dump(self.policy, file)

    def calculate_best_action(self):
        """
        Calculates best action using pre-trained model and pre-calculated policy.

        Returns:
        (NDArray): A pair of best action coordinates.

        """
        best_action = self.board.get_random_action()

        Q_sa_best = self._get_Q(self.board.state, best_action)

        if len(self.board.available_actions) == 1:
            return best_action

        # find action with larger Q
        for action in self.board.available_actions:
            Q_sa = self._get_Q(self.board.state, action)

            if Q_sa > Q_sa_best:
                best_action = action
                Q_sa_best = Q_sa

        return best_action
