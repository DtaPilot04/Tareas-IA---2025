package ejercicio8Puzzle;

import java.util.*;

class PuzzleSolver {
    public static void solve(int[][] initialBoard) {
        PriorityQueue<PuzzleState> openList = new PriorityQueue<>(Comparator.comparingInt(a -> a.cost));
        Set<PuzzleState> closedList = new HashSet<>();

        PuzzleState initialState = new PuzzleState(initialBoard, 2, 2, 0, null);
        openList.add(initialState);

        while (!openList.isEmpty()) {
            PuzzleState currentState = openList.poll();

            if (currentState.isGoal()) {
                printSolution(currentState);
                return;
            }

            closedList.add(currentState);

            for (PuzzleState child : currentState.generateChildren()) {
                if (!closedList.contains(child)) {
                    openList.add(child);
                }
            }
        }
        System.out.println("No solution found.");
    }

    private static void printSolution(PuzzleState state) {
        List<PuzzleState> path = new ArrayList<>();
        while (state != null) {
            path.add(state);
            state = state.parent;
        }
        Collections.reverse(path);

        for (PuzzleState s : path) {
            printBoard(s.board);
            System.out.println();
        }
    }

    private static void printBoard(int[][] board) {
        for (int[] row : board) {
            for (int num : row) {
                System.out.print(num + " ");
            }
            System.out.println();
        }
    }
}