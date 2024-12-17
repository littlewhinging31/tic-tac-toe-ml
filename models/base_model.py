from abc import ABC, abstractmethod

from board import Board


class BaseModel(ABC):
    """
    Abstract base model.
    """

    @abstractmethod
    def __init__(self, board: Board):
        self.board=board

    @abstractmethod
    def train(self):
        """Trains model."""
        pass

    @abstractmethod
    def calculate_best_action(self):
        """Calculates best action while playing a game."""
        pass