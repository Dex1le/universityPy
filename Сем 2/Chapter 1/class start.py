class DataBase:
    pk = 1
    title = 'Классы и объекты'
    author = 'Сергей Балакирев'
    views = 14356
    comments = 12


class Goods:
    title = 'qwerty'
    weight = 154
    tp = 'Еда'
    price = 1024

Goods.price = 2048
Goods.inflation = 100


class Car:
    pass

setattr(Car, 'model', "Тойота")
setattr(Car, 'color', "Розовый")
setattr(Car, 'number', "П111УУ77")


class Notes:
    uid = 1005435
    title = 'Шутка'
    author = 'И.С.Бах'
    pages = 2

print(getattr(Notes, 'author'))


class Dictionary:
    rus = 'Питон'
    eng = 'Python'

print(getattr(Dictionary, 'rus_word', False))


class TravelBlog:
    total_blogs = 0

tb1 = TravelBlog()
tb1.name = 'Франция'
tb1.days = 6

TravelBlog.total_blogs += 1

tb2 = TravelBlog()
tb2.name = 'Италия'
tb2.days = 5

TravelBlog.total_blogs += 1

print(TravelBlog.total_blogs, tb1.days, tb1.name)


class Figure:
    type_fig = 'ellipse'
    color = 'red'

fig1 = Figure()
fig1.start_pt = (10, 5)
fig1.end_pt = (100, 20)
fig1.color = 'blue'

delattr(fig1, 'color')

print(fig1.__dict__)


class Person:
    name = 'Сергей Бакалавр'
    job = 'Programmist'
    city = 'Moscow'

p1 = Person()
print(hasattr(p1, 'job'))