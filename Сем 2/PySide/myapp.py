from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from random import choice
import sys
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QWidget,
    QVBoxLayout
)

window_titles = [
    'My app',
    'Still my app',
    'Patch app'
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My App")

        vbox = QVBoxLayout()

        # Text
        widget = QLabel("QBtorrent")
        font = widget.font()
        font.setPointSize(30)
        widget.setFont(font)
        widget.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        vbox.addWidget(widget)

        # Checkbox
        self.widget1 = QCheckBox('Здесь проверяется патч')

        self.widget1.setCheckState(Qt.CheckState.PartiallyChecked)
        self.widget1.setTristate(True)
        self.widget1.stateChanged.connect(self.show_state)

        vbox.addWidget(self.widget1)

        # ComboBox
        combobox = QComboBox()
        combobox.addItems(['Patch date 27.05.2022', 'Patch date 02.08.2023', 'Patch date 15.11.2024'])

        combobox.currentIndexChanged.connect(self.index_changed)
        combobox.currentTextChanged.connect(self.text_changed)

        vbox.addWidget(combobox)

        # First button
        self.button = QPushButton("Патч", self)
        self.button.setCheckable(False)
        self.button.clicked.connect(self.the_button_was_clicked)
        vbox.addWidget(self.button)

        # Подключение кнопки к действию смены заголовка окна
        self.windowTitleChanged.connect(self.the_window_title_changed)

        # Second button
        button1 = QPushButton("On/Off", self)
        button1.setCheckable(True)
        button1.clicked.connect(self.the_button_was_toggled)
        vbox.addWidget(button1)

        # Создание слоёв с кнопками для главного окна
        main_w = QWidget()
        main_w.setLayout(vbox)
        self.setCentralWidget(main_w)

        # Создание слоя с полем ввода
        self.label = QLabel()

        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)

        vbox.addWidget(self.label)
        vbox.addWidget(self.input)

        container = QWidget()
        container.setLayout(vbox)

        self.setCentralWidget(container)

    def show_state(self):
        if self.widget1.checkState() == Qt.CheckState.Unchecked:
            print("Ни один патч не готов")

        elif self.widget1.checkState() == Qt.CheckState.PartiallyChecked:
            print("Есть незаконченные патчи")

        elif self.widget1.checkState() == Qt.CheckState.Checked:
            print("Все патчи готовы")

    def index_changed(self, i):
        print(i)

    def text_changed(self, s):
        if s == 'Patch date 27.05.2022':
            print('Требуется доработка')

        elif s == 'Patch date 02.08.2023':
            print('Готов к публикации')

        else:
            print('Не готов')

    def the_button_was_clicked(self):
        print("Новый патч")
        new_window_title = choice(window_titles)
        self.setWindowTitle(new_window_title)

    def the_button_was_toggled(self, checked):
        if checked:
            print("Патч включен")
        else:
            print("Патч отключен")

    def the_window_title_changed(self, window_title):
        print("Window title changed: %s" % window_title)

        if window_title == "Patch app":
            self.button.setDisabled(True)


class SecondWindow(QMainWindow, QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Widgets")

        xbox = QVBoxLayout()

        # Text
        widget1 = QLabel("Тупо Виджеты")
        font = widget1.font()
        font.setPointSize(30)
        widget1.setFont(font)
        widget1.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        xbox.addWidget(widget1)
        self.setCentralWidget(widget1)

        # Добавление виджетов
        widgets = [
            QCheckBox,
            QComboBox,
            QDateEdit,
            QDateTimeEdit,
            QDial,
            QDoubleSpinBox,
            QFontComboBox,
            QLabel,
            QLCDNumber,
            QLineEdit,
            QProgressBar,
            QPushButton,
            QRadioButton,
            QSlider,
            QSpinBox,
            QTimeEdit
        ]

        for w in widgets:
            xbox.addWidget(w())

        widget = QWidget()
        widget.setLayout(xbox)
        self.setCentralWidget(widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    window2 = SecondWindow()
    window2.show()

    app.exec()
