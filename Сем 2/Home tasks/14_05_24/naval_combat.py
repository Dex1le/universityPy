from random import randint


class Ship:    # Класс для представления кораблей
    def __init__(self, length, tp=1, x=None, y=None):    # Конструктор
        self.x = x
        self.y = y
        self.length = length
        self.tp = tp
        self._x = x
        self._y = y
        self._length = length
        self._tp = tp
        self._is_move = True
        self._cells = [1] * length

    def set_start_coords(self, x, y):    # Установка начальных координат
        self._x = x
        self._y = y

    def get_start_coords(self):    # Получение начальных координат корабля
        return self._x, self._y

    def move(self, go):    # Перемещение корабля в направлении его ориентации
        if self._is_move:
            if self._tp == 1:
                if self._x is not None:
                    self._x += go
            else:
                if self._y is not None:
                    self._y += go

    def is_collide(self, ship):     # Проверка на столкновение с другим кораблем ship
        if (self._x is not None) and (self._y is not None) and (ship._x is not None) and (ship._y is not None):
            if abs(self._x - ship._x) <= 1 and abs(self._y - ship._y) <= 1:
                return True
        return False

    def is_out_pole(self, size):    # Проверка на выход корабля за пределы игрового поля
        if self._tp == 1 and None not in (self._x, self._length):  # Горизонтальное расположение
            if self._x < 0 or self._x + self._length > size or self._y < 0 or self._y >= size:
                return True
        elif self._tp == 2 and None not in (self._y, self._length):  # Вертикальное расположение
            if self._y < 0 or self._y + self._length > size or self._x < 0 or self._x >= size:
                return True
        return False

    # Магические методы обеспечивающие доступ к коллекции _cells

    def __getitem__(self, index):
        return self._cells[index]

    def __setitem__(self, index, value):
        self._cells[index] = value


class GamePole:      # Класс для описания игрового поля
    def __init__(self, size):
        self._size = size
        self._ships = []

    def init(self):      # Начальная инициализация игрового поля
        ship_count = {1: 4, 2: 3, 3: 2, 4: 1}
        for length, count in ship_count.items():
            for _ in range(count):
                new_ship = Ship(length, tp=randint(1, 2))
                self._ships.append(new_ship)

    def get_ships(self):      # Возвращает коллекцию _ships
        return self._ships

    def move_ships(self):     # Перемещает каждый корабль из коллекции _ships на одну клетку
        for ship in self._ships:
            go = randint(-1, 1)
            ship.move(go)

    def show(self):     # Отображение игрового поля в консоли
        for i in range(self._size):
            for j in range(self._size):
                cell_value = 0
                for ship in self._ships:
                    if (i, j) in ship._cells:
                        cell_value = ship._cells[(i, j)]
                        break
                print(cell_value, end=' ')
            print()

    def get_pole(self):     # Получение текущего игрового поля в виде двухмерного вложенного кортежа
        pole = [[0 for _ in range(self._size)] for _ in range(self._size)]

        for ship in self._ships:
            for x in range(len(ship._cells)):
                y = ship._cells[x]
                pole[x][y] = 1

        return tuple(tuple(row) for row in pole)


# Tests
ship = Ship(2)
ship = Ship(2, 1)
ship = Ship(3, 2, 0, 0)

assert ship._length == 3 and ship._tp == 2 and ship._x == 0 and ship._y == 0, "неверные значения атрибутов объекта класса Ship"
assert ship._cells == [1, 1, 1], "неверный список _cells"
assert ship._is_move, "неверное значение атрибута _is_move"

ship.set_start_coords(1, 2)
assert ship._x == 1 and ship._y == 2, "неверно отработал метод set_start_coords()"
assert ship.get_start_coords() == (1, 2), "неверно отработал метод get_start_coords()"

ship.move(1)
s1 = Ship(4, 1, 0, 0)
s2 = Ship(3, 2, 0, 0)
s3 = Ship(3, 2, 0, 2)
assert s1.is_collide(s2), "неверно работает метод is_collide() для кораблей Ship(4, 1, 0, 0) и Ship(3, 2, 0, 0)"
assert s1.is_collide(
    s3) == False, "неверно работает метод is_collide() для кораблей Ship(4, 1, 0, 0) и Ship(3, 2, 0, 2)"

s2 = Ship(3, 2, 1, 1)
assert s1.is_collide(s2), "неверно работает метод is_collide() для кораблей Ship(4, 1, 0, 0) и Ship(3, 2, 1, 1)"
s2 = Ship(3, 1, 8, 1)
assert s2.is_out_pole(10), "неверно работает метод is_out_pole() для корабля Ship(3, 1, 8, 1)"
s2 = Ship(3, 2, 1, 5)
assert s2.is_out_pole(10) == False, "неверно работает метод is_out_pole(10) для корабля Ship(3, 2, 1, 5)"
s2[0] = 2
assert s2[0] == 2, "неверно работает обращение ship[indx]"
p = GamePole(10)
p.init()
for nn in range(5):
    for s in p._ships:
        assert s.is_out_pole(10) == False, "корабли выходят за пределы игрового поля"
        for ship in p.get_ships():
            if s != ship:
                assert s.is_collide(ship) == False, "корабли на игровом поле соприкасаются"
    p.move_ships()

gp = p.get_pole()
assert type(gp) == tuple and type(gp[0]) == tuple, "метод get_pole должен возвращать двумерный кортеж"
assert len(gp) == 10 and len(gp[0]) == 10, "неверные размеры игрового поля, которое вернул метод get_pole"
pole_size_8 = GamePole(8)
pole_size_8.init()
print("\n Passed")