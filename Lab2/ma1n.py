# Импорт необходимых пакетов
import copy
import time

# Функция для вывода решений и записи их в файл
def show_solutions(solutions: list, startTime) -> None:
    print("Всего решений:", len(solutions))

    # Запись решений в файл
    with open("output.txt", "w") as output_file:
        if not solutions:
            output_file.write("no solutions")
        else:
            for solution in solutions:
                output_file.write(" ".join([str(elem) for elem in solution]) + "\n")

    endTime = time.time()
    print("Программа выполнилась за:", endTime - startTime)

# Функция для выполнения хода
def make_a_move(board: list, N: int, row: int, col: int, solutions: list) -> None:
    solutions.append((row, col))
    board[row][col] = "#"

    # Возможные ходы
    moves = [
        (row + 1, col + 1),
        (row - 1, col - 1),
        (row + 1, col - 1),
        (row - 1, col + 1),
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
        (row + 3, col + 1),
        (row + 3, col - 1),
        (row - 3, col + 1),
        (row - 3, col - 1),
        (row + 1, col + 3),
        (row + 1, col - 3),
        (row - 1, col + 3),
        (row - 1, col - 3),
    ]

    # Помечаем ячейки, на которые нельзя ставить фигуры
    for row_index, col_index in moves:
        if (
            0 <= row_index < N
            and 0 <= col_index < N
            and board[row_index][col_index] != "#"
        ):
            board[row_index][col_index] = "*"

# Основная функция для решения задачи
def solve(
    L: int, N: int, board: list, solutions: list, allSolutions: list, startTime: list
):
    backtrack(L, N, board, 0, -1, solutions, allSolutions)

    show_solutions(allSolutions, startTime)

# Рекурсивная функция для обратного отслеживания
def backtrack(
    L: int,
    N: int,
    board: list,
    row: int,
    col: int,
    solutions: list,
    allSolutions: list,
) -> None:
    while True:
        col += 1

        if col >= N:
            col = 0
            row += 1

        if row >= N:
            break

        if not board[row][col] == "0":
            continue

        now_board: list = copy.deepcopy(board)
        now_solutions: list = copy.deepcopy(solutions)

        make_a_move(now_board, N, row, col, now_solutions)

        if L - 1 == 0:
            allSolutions.append(now_solutions)
            if len(allSolutions) == 1:
                for i_row in board:
                    row_of_board = " ".join(i_row)
                    print(row_of_board)
            continue

        backtrack(
            L - 1,
            N,
            now_board,
            row,
            col,
            now_solutions,
            allSolutions,
        )

# Инициализация данных
def init_data():
    startTime = time.time()

    solutions: list = []
    allSolutions: list = []

    with open("input.txt", "r") as input_file:
        N, L, K = map(int, input_file.readline().split())

        board: list = [["0"] * N for _ in range(N)]

        for _ in range(K):
            row, col = map(int, input_file.readline().split())
            make_a_move(board, N, row, col, solutions)

    return startTime, board, solutions, allSolutions, N, L, K

# Основная функция
def main():
    startTime, board, solutions, allSolutions, N, L, K = init_data()

    print("Размер доски:", N, "Фигур стоит:", K, "Нужно разместить фигур:", L)

    if L == 0:
        if not (len(solutions) == 0):
            allSolutions.append(solutions)
        for row in board:
            row_of_board = " ".join(row)
            print(row_of_board)
        show_solutions(allSolutions, startTime)
        return

    solve(L, N, board, solutions, allSolutions, startTime)

# Выполнение основной функции
if __name__ == "__main__":
    main()
