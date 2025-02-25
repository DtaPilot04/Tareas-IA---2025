package ejercicio8Puzzle;

public class EightPuzzle {
    public static void main(String[] args) {
        int[][] initialBoard = {
            {1, 2, 3},
            {5, 6, 0},
            {7, 8, 4}
        };

        PuzzleSolver.solve(initialBoard);
    }
}