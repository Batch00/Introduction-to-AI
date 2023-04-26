import random
import copy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        self.phase = ' '
        self.player = ' '
        self.total_moves = 0

    def determine_phase(self, board_state):
        num_r = 0
        num_b = 0
        for i in range(5):
            for j in range(5):
                if board_state[i][j] == 'r':
                    num_r += 1
                elif board_state[i][j] == 'b':
                    num_b += 1

        # determine Phase
        if num_r + num_b < 8:
            self.phase = "DROP"
        else:
            self.phase = "MOVE"

        # determine Player
        if num_r + num_b == 0:
            self.player = 'b'
        elif num_r > num_b:
            self.player = 'b'
        elif num_r < num_b:
            self.player = 'r'
        elif self.total_moves % 2 == 0:
            self.player = 'b'
        elif self.total_moves % 2 != 0:
            self.player = 'r'

        return self.phase, self.player
    
    
    def get_opponent(self, piece):
        # Returns the opposing player's piece color given the current player's piece color.
        if piece == 'r':
            return 'b'
        elif piece == 'b':
            return 'r'
        else:
            raise ValueError('Invalid piece color: {}'.format(piece))
        
    def get_2x2_boxes(self, board, player):
        #Returns a list of all 2x2 boxes containing 3 pieces of the same player.
        boxes = []
        for i in range(4):
            for j in range(4):
                if board[i][j] == board[i+1][j] == board[i][j+1] == board[i+1][j+1] == player:
                    boxes.append((i, j))
        return boxes
    
    def get_adjacent_positions(self, row, col):
        #Given a position (row, col) on the Teeko board, returns a list of all adjacent positions.
        adjacent_positions = [(row-1, col), (row+1, col), #left, right
                              (row, col-1), (row, col+1), #Down, Up
                              (row-1, col-1), (row-1, col+1), # diagnals left
                              (row+1, col-1), (row+1, col+1)] # diagnals right
        return [(r, c) for r, c in adjacent_positions if 0 <= r < 5 and 0 <= c < 5]

    def succ(self, state, opponent = False):
        board = state
        phase, player = self.determine_phase(state)

        successors = []

        if opponent == True:
            player = self.get_opponent(player)

        # drop phase
        if phase == "DROP":
            for i in range(5):
                for j in range(5):
                    if board[i][j] == ' ':
                        new_board = copy.deepcopy(board)
                        new_board[i][j] = player
                        new_phase = "MOVE" if len(self.get_2x2_boxes(new_board, player)) > 0 else "DROP"
                        successors.append((new_board, new_phase, self.get_opponent(player), (i,j)))
            return successors
        
        # move phase
        else:
            for i in range(5):
                for j in range(5):
                    if board[i][j] == player:
                        for ni, nj in self.get_adjacent_positions(i, j):
                            if board[ni][nj] == ' ':
                                new_board = copy.deepcopy(board)
                                new_board[ni][nj] = player
                                new_board[i][j] = ' '
                                new_phase = "WIN" if self.game_value(new_board) != 0 else "MOVE"
                                successors.append((new_board, new_phase, self.get_opponent(player), (i,j), (ni,nj)))
            return successors
            
    def MiniMax(self, state, d, max_player = True):
        phase, player = self.determine_phase(state)

        if self.game_value(state) == 1 or self.game_value(state) == -1:
            return self.game_value(state)
        elif d == 0:
            return self.heuristic_game_value(state)
        
        elif max_player:
            value = float('-inf')
            for successor in self.succ(state):
                value = max(value, self.MiniMax(successor[0], d-1, False))
                return value
        else:
            value = float('inf')
            for successor in self.succ(state):
                value = min(value, self.MiniMax(successor[0], d-1, True))
                return value
    
    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        self.total_moves += 1
        phase, player = self.determine_phase(state)
        depth = 5 # set depth to 3 for now

        drop_phase = True

        if phase == "MOVE":
            drop_phase = False   # TODO: detect drop phase

        if drop_phase == False:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            ai_pieces = []
            for row in range(5):
                for col in range(5):
                    if state[row][col] == self.my_piece:
                        ai_pieces.append((row, col))
            
            best_move = None
            best_score = float('-inf')
            for piece in ai_pieces:
                moves = self.succ(state) # get all possible moves
                for move in moves:
                    score = self.MiniMax(move[0], depth, False)
                    if score > best_score:
                        if move[-1] in self.get_adjacent_positions(piece[0], piece[1]):
                            best_score = score
                            best_move = [move[-1], piece]

            board_copy = copy.deepcopy(state)
            board_copy[best_move[1][0]][best_move[1][1]] = ' '
            return best_move

        # TODO: implement a minimax algorithm to play better
        
        best_move = None
        best_score = float('-inf')
        moves = self.succ(state) # get all possible moves
        for move in moves:
            score = self.MiniMax(move[0], depth, False)
            if score > best_score:
                best_score = score
                best_move = [move[-1]]

        return best_move


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """

        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for i in range(2):
            if state[i][i] != ' ' and state[i][i] == state[i+1][i+1] == state[i+2][i+2] == state[i+3][i+3]:
                return 1 if state[i][i]==self.my_piece else -1

        # check / diagonal wins
        for i in range(2):
            j = 3 - i
            if state[i][j] != ' ' and state[i][j] == state[i+1][j-1] == state[i+2][j-2] == state[i+3][j-3]:
                return 1 if state[i][j]==self.my_piece else -1

        # check box wins
        for i in range(4):
            for j in range(4):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j] == state[i][j+1] == state[i+1][j+1]:
                    return 1 if state[i][j]==self.my_piece else -1

        return 0 # no winner yet
    
    def heuristic_game_value(self, state):
        
        # Check if game is in terminal state
        value = self.game_value(state)
        if value != 0:
            return value

        # Evaluate non-terminal states heuristically
        # Number of pieces on the board
        num_pieces = 0
        for row in state:
            for col in row:
                if col != ' ':
                    num_pieces += 1
        piece_ratio = num_pieces / 16

        # Distance between pieces
        distances = []
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece:
                    for k in range(i, 5):
                        for l in range(5):
                            if state[k][l] == self.my_piece:
                                distances.append(abs(i - k) + abs(j - l))
        avg_distance = sum(distances) / len(distances) if distances else 0

        # Combine the above factors into a heuristic value
        heuristic_value = piece_ratio * 0.4 + (1 - piece_ratio) * 0.6 * (1 - avg_distance / 4)

        return heuristic_value

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
