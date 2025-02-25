package ejercicio8Puzzle;

import java.util.*;

class PuzzleState {
    int[][] board;
    int zeroRow, zeroCol;
    int cost;
    int level;
    PuzzleState parent;

    public PuzzleState(int[][] board, int zeroRow, int zeroCol, int level, PuzzleState parent) {
        this.board = new int[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                this.board[i][j] = board[i][j];
            }
        }
        this.zeroRow = zeroRow;
        this.zeroCol = zeroCol;
        this.level = level;
        this.parent = parent;
        this.cost = calculateCost();
    }

    private int calculateCost() {
        int totalCost = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board[i][j] != 0) {
                    int targetRow = (board[i][j] - 1) / 3;
                    int targetCol = (board[i][j] - 1) % 3;
                    totalCost += Math.abs(i - targetRow) + Math.abs(j - targetCol);
                }
            }
        }
        return totalCost + level;
    }

    public boolean isGoal() {
        int[][] goal = {{1, 2, 3}, {4, 5, 6}, {7, 8, 0}};
        return Arrays.deepEquals(board, goal);
    }

    public List<PuzzleState> generateChildren() {
        List<PuzzleState> children = new ArrayList<>();
        int[] rowOffsets = {-1, 0, 1, 0};
        int[] colOffsets = {0, 1, 0, -1};

        for (int i = 0; i < 4; i++) {
            int newRow = zeroRow + rowOffsets[i];
            int newCol = zeroCol + colOffsets[i];

            if (newRow >= 0 && newRow < 3 && newCol >= 0 && newCol < 3) {
                int[][] newBoard = new int[3][3];
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        newBoard[r][c] = board[r][c];
                    }
                }
                newBoard[zeroRow][zeroCol] = newBoard[newRow][newCol];
                newBoard[newRow][newCol] = 0;
                children.add(new PuzzleState(newBoard, newRow, newCol, level + 1, this));
            }
        }
        return children;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        PuzzleState other = (PuzzleState) obj;
        return Arrays.deepEquals(board, other.board);
    }

    @Override
    public int hashCode() {
        return Arrays.deepHashCode(board);
    }
}