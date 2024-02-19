class MediaPlayer:
    def open(self, file):
        self.filename = file

    def play(self):
        print(f"Воспроизведение {self.filename}")

media1 = MediaPlayer()
media2 = MediaPlayer()

media1.open("filemedia1")
media2.open("filemedia2")

media1.play()
media2.play()


class Graph:
    LIMIT_Y = (0, 10)

    def set_data(self, data):
        self.data = data

    def draw(self):
        filtered_data = [str(num) for num in self.data if self.LIMIT_Y[0] <= num <= self.LIMIT_Y[1]]
        print(" ".join(filtered_data))

graph_1 = Graph()
graph_1.set_data([10, -5, 100, 20, 0, 80, 45, 2, 5, 7])
graph_1.draw()


class StreamData:
    def create(self, fields, lst_values):
        if len(fields) != len(lst_values):
            return False

        for field, value in zip(fields, lst_values):
            setattr(self, field, value)

        return True


class Database:
    lst_data = []

    FIELDS = ['id', 'name', 'old', 'salary']

    def select(self, a, b):
        return self.lst_data[a:b+1]

    def insert(self, data):
        for item in data:
            record = {field: value for field, value in zip(self.FIELDS, item.split())}
            self.lst_data.append(record)


class Translator:
    def __init__(self):
        self.translations = {}

    def add(self, eng, rus):
        if eng not in self.translations:
            self.translations[eng] = [rus]
        else:
            if rus not in self.translations[eng]:
                self.translations[eng].append(rus)

    def remove(self, eng):
        if eng in self.translations:
            del self.translations[eng]

    def translate(self, eng):
        return self.translations.get(eng, [])

# Создание экземпляра класса Translator
tr = Translator()

# Добавление связок
tr.add('tree', 'дерево')
tr.add('car', 'машина')
tr.add('car', 'автомобиль')
tr.add('leaf', 'лист')
tr.add('river', 'река')
tr.add('go', 'идти')
tr.add('go', 'ехать')
tr.add('go', 'ходить')
tr.add('milk', 'молоко')

# Удаление связки для слова 'car'
tr.remove('car')

# Перевод слова 'go'
translation = tr.translate('go')

# Вывод русских слов, связанных со словом 'go'
print(" ".join(translation))