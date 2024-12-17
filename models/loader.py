from abc import ABC, abstractmethod

from board import Board
from models.reinforcement_learning.reinforcement_learning import ReinforcementLearningModel


class BaseModelLoader(ABC):
    """
    Abstract base model loader.
    """

    @abstractmethod
    def load_model(self, board: Board):
        """
        Initializes and loads requested model.
        """
        pass


class ReinforcementLearningModelLoader(BaseModelLoader):
    """
    Reinforcement Learning model loader.
    """

    def load_model(self, board: Board = None):
        """
        Initializes and loads Reinforcement Learning model.
        """
        return ReinforcementLearningModel(board)
