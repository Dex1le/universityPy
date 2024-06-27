import time
from typing import List, Tuple, Set
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout, QLineEdit, QHBoxLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent


class Board:
    # Класс для представления шахматной доски
    def __init__(self, size: int):  # Инициализация доски заданного размера
        self.size = size
        self.grid = [['0'] * size for _ in range(size)]

    def update_board(self, row: int, col: int, char: str):  # Обновление клетки доски символом char
        self.grid[row][col] = char

    def get_board(self) -> List[List[str]]:  # Получение текущего состояния доски
        return self.grid

    def display(self):  # Отображение доски в консоли
        for row in self.grid:
            print(" ".join(row))


class Game:
    # Класс для представления игры и её логики
    def __init__(self, board_size: int, num_figures: int, initial_positions: List[Tuple[int, int]]):  # Инициализация игры
        self.board = Board(board_size)
        self.num_figures = num_figures
        self.initial_positions = initial_positions
        self.user_positions = []
        self.all_solutions = []

        for row, col in initial_positions:
            self.make_a_move(row, col)

    def make_a_move(self, row: int, col: int):  # Установка фигуры на доску и пометка запрещенных клеток
        self.board.update_board(row, col, '♔')
        self.mark_invalid_moves(row, col)
        self

    def mark_invalid_moves(self, row: int, col: int):  # Пометка клеток, на которые нельзя ставить фигуры
        moves = self.possible_moves(row, col)
        for r, c in moves:
            if 0 <= r < self.board.size and 0 <= c < self.board.size and self.board.get_board()[r][c] == '0':
                self.board.update_board(r, c, '*')

    def clear_invalid_moves(self):  # Очистка всех помеченных клеток
        for i in range(self.board.size):
            for j in range(self.board.size):
                if self.board.get_board()[i][j] == '*':
                    self.board.update_board(i, j, '0')

    def is_valid_move(self, row: int, col: int, solutions: List[Tuple[int, int]]) -> bool:  # Проверка, можно ли установить фигуру на данную клетку
        if self.board.get_board()[row][col] != '0':
            return False

        moves = self.possible_moves(row, col)
        for r, c in moves:
            if 0 <= r < self.board.size and 0 <= c < self.board.size:
                if (r, c) in solutions or self.board.get_board()[r][c] == '♔':
                    return False
        return True

    def possible_moves(self, row: int, col: int) -> Set[Tuple[int, int]]:  # Получение списка возможных ходов для фигуры
        moves = {
            (row + 1, col + 1), (row - 1, col - 1),
            (row + 1, col - 1), (row - 1, col + 1),
            (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1),
            (row + 3, col + 1), (row + 3, col - 1),
            (row - 3, col + 1), (row - 3, col - 1),
            (row + 1, col + 3), (row + 1, col - 3),
            (row - 1, col + 3), (row - 1, col - 3)
        }
        return moves

    def solve(self):  # Решение задачи размещения фигур
        self.backtrack(self.num_figures, 0, -1, self.initial_positions + self.user_positions)

    def backtrack(self, L: int, row: int, col: int, solutions: List[Tuple[int, int]]):
        # Рекурсивный метод для нахождения всех возможных решений
        if L == 0:
            self.all_solutions.append(solutions.copy())
            return

        for r in range(row, self.board.size):
            start_col = col + 1 if r == row else 0
            for c in range(start_col, self.board.size):
                if (r, c) not in solutions and self.is_valid_move(r, c, solutions):
                    solutions.append((r, c))
                    self.backtrack(L - 1, r, c, solutions)
                    solutions.pop()

    def show_solutions(self):  # Отображение решений в консоли и запись их в файл
        if not self.all_solutions:
            print("No solutions found.")
            return

        print("Всего решений:", len(self.all_solutions))
        print("Первое решение:")
        first_solution = self.all_solutions[0]
        for row, col in first_solution:
            self.board.update_board(row, col, '♔')
            self.mark_invalid_moves(row, col)
        self.board.display()

        # Записываем все решения в файл
        with open("output.txt", "w") as output_file:
            for solution in self.all_solutions:
                solution_str = " ".join([f"({row},{col})" for row, col in solution])
                output_file.write(solution_str + "\n")

    def create_solution_board(self, solution: List[Tuple[int, int]]) -> List[List[str]]:  # Создание доски с указанным решением
        board = Board(self.board.size)
        for row, col in self.initial_positions:
            board.update_board(row, col, '♔')
            self.mark_invalid_moves(row, col)
        for row, col in solution:
            board.update_board(row, col, '♔')
            self.mark_invalid_moves(row, col)
        return board.get_board()


class MainWindow(QMainWindow):
    # Класс главного окна приложения с графическим интерфейсом
    def __init__(self):  # Инициализация главного окна
        super().__init__()
        self.setWindowTitle("Ввод данных")
        self.setGeometry(100, 100, 300, 200)

        self.init_ui()

    def init_ui(self):  # Инициализация графического интерфейса
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        size_layout = QHBoxLayout()
        size_label = QLabel("Размер доски (N):")
        self.size_input = QLineEdit()
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_input)

        figures_layout = QHBoxLayout()
        figures_label = QLabel("Количество фигур (L):")
        self.figures_input = QLineEdit()
        figures_layout.addWidget(figures_label)
        figures_layout.addWidget(self.figures_input)

        start_button = QPushButton("Начать")
        start_button.clicked.connect(self.start_game)

        layout.addLayout(size_layout)
        layout.addLayout(figures_layout)
        layout.addWidget(start_button)

        central_widget.setLayout(layout)

    def start_game(self):  # Обработка нажатия кнопки "Начать" для создания нового окна с игрой
        try:
            board_size = int(self.size_input.text())
            num_figures = int(self.figures_input.text())
        except ValueError:
            print("Пожалуйста, введите корректные числовые значения.")
            return

        initial_positions = []  # Начальные позиции фигур могут быть пустыми или заданы пользователем

        self.game_window = GameWindow(board_size, num_figures, initial_positions)
        self.game_window.show()


class GameWindow(QMainWindow):
    # Класс окна игры с графическим интерфейсом
    def __init__(self, board_size: int, num_figures: int, initial_positions: List[Tuple[int, int]]):  # Инициализация окна игры
        super().__init__()
        self.game = Game(board_size, num_figures, initial_positions)
        self.setWindowTitle("Game Solver")
        self.setGeometry(100, 100, 600, 600)

        self.init_ui()

    def init_ui(self):  # Инициализация графического интерфейса
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.board_layout = QGridLayout()

        self.update_board_ui()

        solve_button = QPushButton("Решение")
        solve_button.clicked.connect(self.solve_game)

        layout.addLayout(self.board_layout)
        layout.addWidget(solve_button)

        central_widget.setLayout(layout)

    def update_board_ui(self):  # Обновление графического интерфейса доски
        for i in range(self.board_layout.count()):
            self.board_layout.itemAt(i).widget().deleteLater()

        board = self.game.board.get_board()
        for i in range(self.game.board.size):
            for j in range(self.game.board.size):
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setFixedSize(50, 50)
                if (i, j) in self.game.user_positions:
                    label.setText('♔')
                    label.setStyleSheet("background-color: red; color: black; border: 1px solid black;")
                elif (i, j) in self.game.initial_positions:
                    label.setText('♔')
                    label.setStyleSheet("background-color: green; color: black; border: 1px solid black;")
                elif board[i][j] == '*':
                    label.setStyleSheet("background-color: red; border: 1px solid black;")
                else:
                    color = 'white' if (i + j) % 2 == 0 else 'black'
                    label.setStyleSheet(f"background-color: {color}; border: 1px solid black;")

                label.installEventFilter(self)
                label.row = i
                label.col = j
                self.board_layout.addWidget(label, i, j)

    def eventFilter(self, source, event):  # Обработка кликов по клеткам доски
        if event.type() == QMouseEvent.MouseButtonPress:
            row, col = source.row, source.col
            if event.button() == Qt.LeftButton:
                if self.game.board.get_board()[row][col] == '0':
                    self.game.make_a_move(row, col)
                    self.game.user_positions.append((row, col))
                elif self.game.board.get_board()[row][col] == '♔' and (row, col) in self.game.user_positions:
                    self.game.board.update_board(row, col, '0')
                    self.game.user_positions.remove((row, col))
                    self.game = Game(self.game.board.size, self.game.num_figures, self.game.initial_positions)
                    for pos in self.game.user_positions:
                        self.game.make_a_move(*pos)
                self.update_board_ui()
            elif event.button() == Qt.RightButton:
                if self.game.board.get_board()[row][col] == '♔' and (row, col) in self.game.user_positions:
                    self.game.board.update_board(row, col, '0')
                    self.game.user_positions.remove((row, col))
                    self.game = Game(self.game.board.size, self.game.num_figures, self.game.initial_positions)
                    for pos in self.game.user_positions:
                        self.game.make_a_move(*pos)
                self.update_board_ui()
        return super().eventFilter(source, event)

    def solve_game(self):  # Обработка нажатия кнопки "Решение" для решения задачи
        self.game.solve()
        if not self.game.all_solutions:
            print("Решений не найдено.")
            self.close()
            return

        # Создаем новое окно для отображения первого решения
        self.solution_window = SolutionWindow(self.game)
        self.solution_window.show()


class SolutionWindow(QMainWindow):
    # Класс окна для отображения решения
    def __init__(self, game: Game):  # Инициализация окна с решением
        super().__init__()
        self.setWindowTitle("Solution")
        self.setGeometry(100, 100, 600, 600)

        self.game = game
        self.solution = self.game.all_solutions[0]

        self.init_ui()

    def init_ui(self):  # Инициализация графического интерфейса
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.board_layout = QGridLayout()

        self.update_board_ui()

        save_button = QPushButton("Вывести решения")
        save_button.clicked.connect(self.save_solutions)

        layout.addLayout(self.board_layout)
        layout.addWidget(save_button)

        central_widget.setLayout(layout)

    def update_board_ui(self):  # Обновление графического интерфейса доски
        for i in range(self.board_layout.count()):
            self.board_layout.itemAt(i).widget().deleteLater()

        board = self.game.create_solution_board(self.solution)
        for i in range(self.game.board.size):
            for j in range(self.game.board.size):
                label = QLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setFixedSize(50, 50)
                if (i, j) in self.game.user_positions:
                    label.setText('♔')
                    label.setStyleSheet("background-color: blue; color: black; border: 1px solid black;")
                elif (i, j) in self.solution:
                    label.setText('♔')
                    label.setStyleSheet("background-color: green; color: black; border: 1px solid black;")
                elif board[i][j] == '*':
                    label.setStyleSheet("background-color: red; border: 1px solid black;")
                else:
                    color = 'white' if (i + j) % 2 == 0 else 'black'
                    label.setStyleSheet(f"background-color: {color}; border: 1px solid black;")

                self.board_layout.addWidget(label, i, j)

    def save_solutions(self):  # Обработка нажатия кнопки "Вывести решения" для записи решений в файл
        self.game.show_solutions()
        print("Решения записаны в файл output.txt")


def main():  # Основная функция программы
    app = QApplication([])

    main_window = MainWindow()
    main_window.show()

    app.exec_()


if __name__ == "__main__":
    main()
