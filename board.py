from enum import Enum

import numpy as np
from numpy.typing import NDArray


class Board:
    """Represents game board."""

    class ValueEnum(Enum):
        X = 1
        O = -1

    def __init__(self):
        """Initializes the empty board."""
        self.state = np.zeros(shape=(3, 3))
        self.available_actions = None

    def make_move(self, action: NDArray[np.int64], value: int):
        """Sets value for a specific position."""
        self.state[action[0], action[1]] = value

    def revert_action(self, action: NDArray[np.int64]):
        """Reverts an action."""
        self.state[action[0], action[1]] = 0

    def calculate_available_actions(self):
        """Calculate available actions."""
        return np.argwhere(self.state == 0)

    def get_random_action(self):
        """Calculates random action from available actions."""
        self.available_actions = self.calculate_available_actions()
        idx = np.random.choice(self.available_actions.shape[0])
        return self.available_actions[idx]

    def is_winner(self, value: int):
        """
        Checks if current player has won a game.

        Parameters:
        value (int): Current player value.

        Returns:

        """
        for row in self.state:
            if np.all(row == value):
                return True

        for column in range(self.state.shape[1]):
            if np.all(self.state[:, column] == value):
                return True

        if self.state[0][0] == self.state[1][1] == self.state[2][2] == value:
            return True

        if self.state[0][2] == self.state[1][1] == self.state[2][0] == value:
            return True

        return False

    def is_game_over(self):
        """Checks if the game is over (no available actions has left)."""
        if not np.any(np.argwhere(self.state == 0)):
            return True

        return False

    def has_winning_move(self, value: int):
        """
        Calculates if player has a winning move.

        Parameters:
        value (int): Current player value.

        Returns:

        """
        available_actions = np.argwhere(self.state == 0)

        for action in available_actions:
            self.state[action[0], action[1]] = value

            if self.is_winner(value):
                self.state[action[0], action[1]] = 0
                return True

            self.state[action[0], action[1]] = 0

        return False

    def print_board(self):
        """Print current state on the board."""
        state = np.where(
            self.state == 1, 'X',
            np.where(
                self.state == -1, 'O',
                np.where(self.state == 0, ' ', self.state)
            )
        )
        for row in state: print(" | ".join(row))
