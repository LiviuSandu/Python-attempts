"""
Defines a class to set chess pieces on a given board so that they can't attack each other.

Identical solutions because of swapped kings are excluded.
Symmetrical solutions are included.
Designed for a specific set of (multiple) pieces.
Can be easily extended to work for any number of kings, queens, rooks, bishops, knights, pawns.

Class: NChessPiecesProblem
    Method: place_pieces
"""

import numpy as np
import sys
import time


class NChessPiecesProblem:
    """Class to perform chess pieces placement on a given table so that they do not attack each other."""

    FREE_CELLS = 0
    BLOCKED_CELLS = 1
    TAKEN_CELLS = 2
    FIRST_KING_ON = 3
    FIRST_KING_CELL = 4
    PIECE_ID = 5

    def __init__(self, pieces, board_rows, board_columns):
        """
        Constructor.

        Input parameters
        ----------------
        pieces: list of strings containing pieces' codes
        board_rows: number of rows on the board (int)
        board_columns: number of columns on the board (int)
        """

        self._chess_pieces = pieces
        self._last_piece = len(self._chess_pieces) - 1
        self._rows = board_rows
        self._columns = board_columns
        self._indices = \
            np.array([i for i in range(0, self._rows * self._columns)])
        self._solutions = 0
        self._masks = None

    def _initialize_state(self):
        """
        Build initial state.

        Returns
        -------
        list: initial state of the algorithm
        """

        state = [None] * 6
        state[NChessPiecesProblem.FREE_CELLS] = self._indices.copy()
        state[NChessPiecesProblem.BLOCKED_CELLS] = \
            np.full(self._rows * self._columns, False)
        state[NChessPiecesProblem.TAKEN_CELLS] = \
            np.full(self._rows * self._columns, False)
        state[NChessPiecesProblem.FIRST_KING_ON] = False
        state[NChessPiecesProblem.FIRST_KING_CELL] = None
        state[NChessPiecesProblem.PIECE_ID] = 0
        return state

    @staticmethod
    def _check_king_placement(rank, cell, state, new_state):
        """
        Define and propagate the state of the kings' placement on board.

        Input parameters
        ----------------
        rank: king's appearance -first or second (int)
        cell: king's position (int)
        state: current state (list)

        Output parameters
        -----------------
        new_state: future state (list)

        Returns
        -------
        bool: True if the king may be placed on the board
        """

        if rank == 0:
            new_state[NChessPiecesProblem.FIRST_KING_CELL] = cell
            new_state[NChessPiecesProblem.FIRST_KING_ON] = True
            return True
        else:
            if cell > state[NChessPiecesProblem.FIRST_KING_CELL]:
                new_state[NChessPiecesProblem.FIRST_KING_ON] = \
                    state[NChessPiecesProblem.FIRST_KING_ON]
                new_state[NChessPiecesProblem.FIRST_KING_CELL] = \
                    state[NChessPiecesProblem.FIRST_KING_CELL]
                return True
            else:
                return False

    def _place_piece_on_board(
                              self, cell, piece, multiple,
                              rank, state, new_state
                             ):
        """
        Define the new board state after placing a new piece.

        Input parameters
        ----------------
        cell: piece position (int)
        piece: piece code (string)
        multiple: multiple pieces of same kind -for kings only (bool)
        rank: king's appearance -first or second (int)
        state: current state (list)

        Output parameters
        -----------------
        new_state: future state (list)

        Returns
        -------
        bool: True if the piece may be placed on the board
        """

        if state[NChessPiecesProblem.BLOCKED_CELLS][cell]:
            return False
        new_blocked_cells = self._masks[(piece, cell)]
        if np.any(new_blocked_cells, where=state[NChessPiecesProblem.TAKEN_CELLS]):
            return False
        if multiple:
            if not self._check_king_placement(rank, cell, state, new_state):
                return False
        new_state[NChessPiecesProblem.BLOCKED_CELLS] = \
            (state[NChessPiecesProblem.BLOCKED_CELLS] | new_blocked_cells).copy()
        new_state[NChessPiecesProblem.TAKEN_CELLS] = \
            state[NChessPiecesProblem.TAKEN_CELLS].copy()
        new_state[NChessPiecesProblem.TAKEN_CELLS][cell] = True
        new_state[NChessPiecesProblem.FREE_CELLS] = \
            self._indices[np.logical_not(new_state[NChessPiecesProblem.BLOCKED_CELLS])]
        new_state[NChessPiecesProblem.PIECE_ID] = \
            state[NChessPiecesProblem.PIECE_ID] + 1
        return True

    def _do_place_pieces(self, state):
        """
        Executive recursive method for piece placement.

        Input parameters
        ----------------
        state: list of state parameters
        """

        piece_id = state[NChessPiecesProblem.PIECE_ID]
        new_piece = self._chess_pieces[piece_id]
        if new_piece == 'K1':
            king = True
            rank = 0
        elif new_piece == 'K2':
            king = True
            rank = 1
        else:
            king = False
            rank = None

        free_cells = state[NChessPiecesProblem.FREE_CELLS]
        for free_cell_index in free_cells:
            new_state = [None] * 6
            if self._place_piece_on_board(
                                          free_cell_index, new_piece,
                                          king, rank, state, new_state
                                         ):
                if piece_id == self._last_piece:
                    self._solutions += 1
                else:
                    self._do_place_pieces(new_state)

    def place_pieces(self):
        """Place pieces on the board so that they do not attack each other.

        Returns
        -------
        int: Number of possible placements
        """

        self._masks = self._make_masks()
        initial_state = self._initialize_state()
        self._do_place_pieces(initial_state)
        return self._solutions

    def _queen_mask(self, i, j):
        """
        Define the mask of fields covered by a queen.

        Input parameters
        ----------------
        i: row position (int)
        j: column position (int)

        Returns
        -------
        numpy 1D array of bool: flattened board representation
        """

        dimension = max(self._rows, self._columns)
        mask = np.zeros((2 * dimension - 1, 2 * dimension - 1))
        mask[dimension - 1, :] = 1
        mask[:, dimension - 1] = 1
        first_diagonal = np.diag(np.full(2 * dimension - 1, 1))
        second_diagonal = np.rot90(first_diagonal)
        mask = mask + first_diagonal + second_diagonal
        mask = np.array(mask, dtype=bool)
        cut = mask[dimension - 1 - i: dimension + self._rows - 1 - i,
                   dimension - 1 - j: dimension + self._columns - 1 - j]
        return cut.flatten()

    def _rook_mask(self, i, j):
        """
        Define the mask of fields covered by a rook.

        Input parameters
        ----------------
        i: row position (int)
        j: column position (int)

        Returns
        -------
        numpy 1D array of bool: flattened board representation
        """

        dimension = max(self._rows, self._columns)
        mask = np.full((2 * dimension - 1, 2 * dimension - 1), False)
        mask[dimension - 1, :] = True
        mask[:, dimension - 1] = True
        cut = mask[dimension - 1 - i: dimension + self._rows - 1 - i,
                   dimension - 1 - j: dimension + self._columns - 1 - j]
        return cut.flatten()

    def _bishop_mask(self, i, j):
        """
        Define the mask of fields covered by a bishop.

        Input parameters
        ----------------
        i: row position (int)
        j: column position (int)

        Returns
        -------
        numpy 1D array of bool: flattened board representation
        """

        dimension = max(self._rows, self._columns)
        mask = np.zeros((2 * dimension - 1, 2 * dimension - 1))
        first_diagonal = np.diag(np.full(2 * dimension - 1, 1))
        second_diagonal = np.rot90(first_diagonal)
        mask = mask + first_diagonal + second_diagonal
        mask = np.array(mask, dtype=bool)
        cut = mask[dimension - 1 - i: dimension + self._rows - 1 - i,
                   dimension - 1 - j: dimension + self._columns - 1 - j]
        return cut.flatten()

    def _king_mask(self, i, j):
        """
        Define the mask of fields covered by a king.

        Input parameters
        ----------------
        i: row position (int)
        j: column position (int)

        Returns
        -------
        numpy 1D array of bool: flattened board representation
        """

        dimension = max(self._rows, self._columns)
        mask = np.full((2 * dimension - 1, 2 * dimension - 1), False)
        mask[dimension - 2: dimension + 1, dimension - 2: dimension + 1] = True
        cut = mask[dimension - 1 - i: dimension + self._rows - 1 - i,
                   dimension - 1 - j: dimension + self._columns - 1 - j]
        return cut.flatten()

    def _knight_mask(self, i, j):
        """
        Define the mask of fields covered by a knight.

        Input parameters
        ----------------
        i: row position (int)
        j: column position (int)

        Returns
        -------
        numpy 1D array of bool: flattened board representation
        """

        dimension = max(self._rows, self._columns)
        mask = np.full((2 * dimension - 1, 2 * dimension - 1), False)
        mask[dimension - 1, dimension - 1] = True
        mask[dimension - 3, dimension - 2] = True
        mask[dimension - 3, dimension] = True
        mask[dimension - 2, dimension - 3] = True
        mask[dimension - 2, dimension + 1] = True
        mask[dimension, dimension - 3] = True
        mask[dimension, dimension + 1] = True
        mask[dimension + 1, dimension - 2] = True
        mask[dimension + 1, dimension] = True
        cut = mask[dimension - 1 - i: dimension + self._rows - 1 - i,
                   dimension - 1 - j: dimension + self._columns - 1 - j]
        return cut.flatten()

    def _make_masks(self):
        """
        Build mask map.

        Returns
        -------
        dictionary:
            key (string, int): (piece_code, piece_position)
            value (list of bool): mask representation
        """

        masks = {}
        masks.update({
                          ('Q', i * self._columns + j):
                          self._queen_mask(i, j) for i in range(0, self._rows) for j in range(0, self._columns)
                     })
        masks.update({
                          ('R', i * self._columns + j):
                          self._rook_mask(i, j) for i in range(0, self._rows) for j in range(0, self._columns)
                     })
        masks.update({
                          ('B', i * self._columns + j):
                          self._bishop_mask(i, j) for i in range(0, self._rows) for j in range(0, self._columns)
                     })
        masks.update({
                          ('K1', i * self._columns + j):
                          self._king_mask(i, j) for i in range(0, self._rows) for j in range(0, self._columns)
                     })
        masks.update({
                          ('K2', i * self._columns + j):
                          self._king_mask(i, j) for i in range(0, self._rows) for j in range(0, self._columns)
                     })
        masks.update({
                          ('N', i * self._columns + j):
                          self._knight_mask(i, j) for i in range(0, self._rows) for j in range(0, self._columns)
                     })
        return masks


def main():
    start = time.time()
    solver = NChessPiecesProblem(('Q', 'R', 'B', 'K1', 'K2', 'N'), 6, 9)
    print(solver.place_pieces())
    end = time.time()
    print(end - start)
    return 0


if __name__ == '__main__':
    sys.exit(main())
