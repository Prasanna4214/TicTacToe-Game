import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.tictactoe_agent import TicTacToeAgent


board = [
    ["X", "O", ""],
    ["", "X", ""],
    ["O", "", ""]
]

agent = TicTacToeAgent()

move = agent.get_best_move(board, "X")

print("Best Move =", move)