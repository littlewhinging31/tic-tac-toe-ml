import re

import numpy as np

from board import Board
from models.base_model import BaseModel


class ConsoleInterface:
    """Represents console interface."""

    def __init__(self):
        self.action_pattern = re.compile(r'^\s*([0-3])\s*,\s*([0-3])\s*$')

    def visualize(self, board: Board, model: BaseModel):
        """Visualize playing a game by computer using pre-trained model."""
        value = board.ValueEnum.X.value

        # make random first move
        action = board.get_random_action()
        board.make_move(action, value)

        print(f"{board.ValueEnum(value)} move:")
        board.print_board()

        value = -value

        while True:
            input("Press any key to proceed:")
            best_action = model.calculate_best_action()
            board.make_move(best_action, value)

            print(f"{board.ValueEnum(value)} move:")
            board.print_board()

            if board.is_winner(value):
                print("The winner is %s" % board.ValueEnum(value))
                break
            elif board.is_game_over():
                print("Draw!")
                break
            else:
                # switch players
                value = -value

    def play(self, board: Board, model: BaseModel):
        """Visualize playing a game by computer against human player using pre-trained model."""
        print("Welcome to Tic-Tac-Toe game powered by Reinforcement Learning!\n")

        user_value = input("To start enter a value (X or O):\n")

        if user_value not in (board.ValueEnum.X.name, board.ValueEnum.O.name):
            while True:
                user_value = input("Incorrect value! Please, try again:")

                if user_value in (board.ValueEnum.X.name, board.ValueEnum.O.name):
                    break

        print("Game started!\n")

        value = board.ValueEnum.X.value

        is_user_active_player = None

        # make a first move
        if user_value == board.ValueEnum.X.name:
            action = input("Make a first move! Enter a position on board (a pair of coordinates, e.g. (0,1), (2,2)):\n")
            match = self.action_pattern.match(action)

            if not match:
                while True:
                    action = input("Incorrect value! Please, try again:")

                    if self.action_pattern.match(action):
                        break

            action = tuple(map(int, action.split(',')))
            board.state[int(action[0]), int(action[1])] = value

            print("Your move:")
            board.print_board()

            is_user_active_player = False
            value = -value
        else:
            action = board.get_random_action()
            board.make_move(action, value)

            print("Opponent's move:")
            board.print_board()

            is_user_active_player = True
            value = -value

        while True:
            if not is_user_active_player:
                best_action = model.calculate_best_action()
                board.make_move(best_action, value)

                print("Opponent's move:")
                board.print_board()

                if board.is_winner(value):
                    print("You loose!")
                    break
                elif board.is_game_over():
                    print("Draw!")
                    break
                else:
                    is_user_active_player = True
                    value = -value
            else:
                action = input("Your turn! Enter a position on board (a pair of coordinates, e.g. (0,1), (2,2)):\n")

                match = self.action_pattern.match(action)

                if not match:
                    while True:
                        action = input("Incorrect value! Please, try again:")

                        if self.action_pattern.match(action):
                            break

                action = tuple(map(int, action.split(',')))

                available_actions = board.calculate_available_actions()

                if not any(np.all(a == action) for a in available_actions):
                    while True:
                        action = input("This move is not available! Please, try again:")
                        action = tuple(map(int, action.split(',')))

                        if any(np.all(a == action) for a in available_actions):
                            break

                board.state[int(action[0]), int(action[1])] = value

                print("Your move:")
                board.print_board()

                if board.is_winner(value):
                    print("You won!")
                    break
                elif board.is_game_over():
                    print("Draw!")
                    break
                else:
                    is_user_active_player = False
                    value = -value
