import argparse

from board import Board
from interface import ConsoleInterface
from models import loader


def main():
    parser = argparse.ArgumentParser(description="Play game or visualize algorithm.")

    parser.add_argument('-p', '--play', action='store_true', help="Play against user.")

    parser.add_argument('-v', '--visualize', action='store_true', help="Visualize algorithm.")

    args = parser.parse_args()

    board = Board()
    interface = ConsoleInterface()

    model = loader.ReinforcementLearningModelLoader().load_model(board=board)
    model.load_policy()

    if args.play:
        interface.play(board, model)

    if args.visualize:
        interface.visualize(board, model)


if __name__ == "__main__":
    main()