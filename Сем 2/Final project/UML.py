import pydot
from IPython.display import Image


# Читаем код из файла
file_path = '/Users/dex1le/PycharmProjects/university/Сем 2/Final project/Chess.py'

with open(file_path, 'r') as file:
    chess_code = file.read()

chess_code[:2000]

# Создаем новый граф
graph = pydot.Dot(graph_type='graph', rankdir='TB')

# Создаем узлы для классов Board, Game, MainWindow, GameWindow и SolutionWindow с упрощенными метками
board_node = pydot.Node("Board", shape="record", label="{Board|size\\lgrid\\l|__init__\\lupdate_board\\lget_board\\ldisplay\\l}")
game_node = pydot.Node("Game", shape="record", label="{Game|board\\lnum_figures\\linitial_positions\\luser_positions\\lall_solutions\\l|__init__\\lmake_a_move\\lmark_invalid_moves\\lclear_invalid_moves\\l}")
main_window_node = pydot.Node("MainWindow", shape="record", label="{MainWindow|game\\lsolution\\lboard_layout\\l|__init__\\lupdate_board_ui\\lsave_solutions\\l}")
game_window_node = pydot.Node("GameWindow", shape="record", label="{GameWindow|game\\lboard_layout\\l|__init__\\lupdate_board_ui\\l}")
solution_window_node = pydot.Node("SolutionWindow", shape="record", label="{SolutionWindow|game\\lsolution\\lboard_layout\\l|__init__\\lupdate_board_ui\\lsave_solutions\\l}")

# Добавляем узлы в граф
graph.add_node(board_node)
graph.add_node(game_node)
graph.add_node(main_window_node)
graph.add_node(game_window_node)
graph.add_node(solution_window_node)

# Создаем связи между классами
graph.add_edge(pydot.Edge(game_node, board_node))
graph.add_edge(pydot.Edge(main_window_node, game_node))
graph.add_edge(pydot.Edge(game_window_node, game_node))
graph.add_edge(pydot.Edge(solution_window_node, game_node))

# Сохраняем граф как изображение в формате JPG
output_path = "Chess_UML.jpg"
graph.write_jpg(output_path)

# Отображаем изображение
Image(output_path)

