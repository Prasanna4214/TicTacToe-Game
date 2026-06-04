def check_winner(board):

    for row in board:
        if row[0] == row[1] == row[2] and row[0] != "":
            return row[0]

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != "":
            return board[0][col]

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != "":
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != "":
        return board[0][2]

    return None


def is_full(board):
    for row in board:
        for cell in row:
            if cell == "":
                return False
    return True


def minimax(board, depth, is_maximizing, ai, opponent):

    winner = check_winner(board)

    if winner == ai:
        return 10 - depth

    if winner == opponent:
        return depth - 10

    if is_full(board):
        return 0

    if is_maximizing:

        best_score = -1000

        for r in range(3):
            for c in range(3):

                if board[r][c] == "":
                    board[r][c] = ai

                    score = minimax(
                        board,
                        depth + 1,
                        False,
                        ai,
                        opponent
                    )

                    board[r][c] = ""

                    best_score = max(best_score, score)

        return best_score

    else:

        best_score = 1000

        for r in range(3):
            for c in range(3):

                if board[r][c] == "":
                    board[r][c] = opponent

                    score = minimax(
                        board,
                        depth + 1,
                        True,
                        ai,
                        opponent
                    )

                    board[r][c] = ""

                    best_score = min(best_score, score)

        return best_score