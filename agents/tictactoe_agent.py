class TicTacToeAgent:

    def get_best_move(self, board, ai):

        # Simple agent for testing

        for r in range(3):
            for c in range(3):
                if board[r][c] == "":
                    return (r, c)

        return None