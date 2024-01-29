import time


# Функция которая задаёт ходы
def poser(row: int, col: int, board: list) -> list:

    king_camel_figure = [
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
        (row - 1, col - 3)
    ]

    board[row][col] = '#'

    # Помечаем ячейки, на которые нельзя ставить фигуры
    for i in king_camel_figure:
        m, n = i[0], i[1]
        if 0 <= m < len(board) and 0 <= n < len(board):
            board[m][n] = '*'
    return board


# Функция которая создает доску
def create_board(N, solutions: list[tuple[int, int]]) -> list[list[str]]:
    board: list = [["0"] * N for _ in range(N)]
    for row, col in solutions:
        poser(row, col, board)
    return board


# Функция возвращающая список ходов
def moves(row: int, col: int) -> list:

    moves = {
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
    }

    return moves


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
def make_a_move(row: int, col: int, board: list) -> None:
    board[row][col] = "#"


# Основная функция для решения задачи
def solve(L: int, N: int, solutions: list, allSolutions: list, startTime: float):

    backtrack(L, N, 0, -1, solutions, allSolutions)

    show_solutions(allSolutions, startTime)


# Рекурсивная функция для обратного отслеживания
def backtrack(
    L: int,
    N: int,
    row: int,
    col: int,
    solutions: list,
    allSolutions: list,
) -> None:
    if L == 0:
        allSolutions.append(solutions.copy())
        if len(allSolutions) == 1:
            for i in create_board(N, solutions):
                print(i)
        return

    for r in range(row, N):
        start_col = col + 1 if r == row else 0
        for c in range(start_col, N):
            if (r, c) not in solutions and not moves(r, c).intersection(solutions):
                solutions.append((r, c))
                backtrack(L - 1, N, r, c, solutions, allSolutions)
                solutions.pop()


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
            solutions.append((row, col))
            make_a_move(row, col, board)

    return startTime, board, solutions, allSolutions, N, L, K


# Основная функция
def main():
    startTime, board, solutions, allSolutions, N, L, K = init_data()

    print("Размер доски:", N, "Нужно разместить фигур:", L, "Фигур стоит:", K)
    if L == 0:
        if not (len(solutions) == 0):
            allSolutions.append(solutions)
        for row in board:
            row_of_board = " ".join(row)
            print(row_of_board)
        show_solutions(allSolutions, startTime)
        return

    solve(L, N, solutions, allSolutions, startTime)


# Выполнение основной функции
if __name__ == "__main__":
    main()