from typing import Union


class Vertex:
    def __init__(self):
        self._links = []

    @property
    def links(self):
        return self._links


class Link:
    def __init__(self, v1, v2, dist=1):
        self._v1 = v1
        self._v2 = v2
        self._dist = dist

    @property
    def v1(self):
        return self._v1

    @property
    def v2(self):
        return self._v2

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, val):
        self._dist = val


class Station(Vertex):
    def __init__(self, name):
        super().__init__()
        self.name = name


class LinkMetro(Link):
    def __init__(self, v1, v2, dist=1):
        super().__init__(v1, v2, dist)


class LinkedGraph:
    def __init__(self):
        self._links = []
        self._vertex = []

    def add_vertex(self, v):
        if v not in self._vertex:
            self._vertex.append(v)

    def add_link(self, link: Link):
        cur_links = list(filter(lambda cur_link:
                                (link.v1 == cur_link.v1 and link.v2 == cur_link.v2) or
                                (link.v1 == cur_link.v2 and link.v2 == cur_link.v1),
                                self._links))
        if not cur_links:
            self._links.append(link)
            link.v1.links.append(link)
            link.v2.links.append(link)

        if link.v1 not in self._vertex:
            self._vertex.append(link.v1)

        if link.v2 not in self._vertex:
            self._vertex.append(link.v2)

    def _find_lowest_cost(self, costs, processed):
        lowest_cost = float('inf')
        lowest_cost_node = None

        for node in costs:
            cost = costs[node]
            if cost < lowest_cost and node not in processed:
                lowest_cost = cost
                lowest_cost_node = node
        return lowest_cost_node

    def find_path(self, start_v: Union[Station, Vertex], stop_v: Vertex):
        costs = {v: float('inf') for v in self._vertex}
        costs[start_v] = 0
        parents = {v: None for v in self._vertex}
        processed = set()

        node = start_v
        while node is not None:
            cost = costs[node]
            neighbors = node.links

            for link in neighbors:
                neighbor = link.v2
                new_cost = cost + link.dist
                if new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    parents[neighbor] = node
            processed.add(node)
            node = self._find_lowest_cost(costs, processed)

        links = []
        vertexes = []
        while stop_v:
            if isinstance(stop_v, Station):
                vertexes.append(stop_v.name)
            else:
                vertexes.append(stop_v)

            parent = parents[stop_v]

            if parent:
                for link in parent.links:
                    if link.v2 is stop_v:
                        links.append(link)
            stop_v = parent

            if stop_v is None:
                break

        final_str = '['
        for v in vertexes[::-1]:
            if isinstance(v, str):
                final_str += v + ', '

        return final_str[:-2] + ']', links[::-1]


# test
map2 = LinkedGraph()
v1 = Vertex()
v2 = Vertex()
v3 = Vertex()
v4 = Vertex()
v5 = Vertex()

map2.add_link(Link(v1, v2))
map2.add_link(Link(v2, v3))
map2.add_link(Link(v2, v4))
map2.add_link(Link(v3, v4))
map2.add_link(Link(v4, v5))

assert len(map2._links) == 5, "неверное число связей в списке _links класса LinkedGraph"
assert len(map2._vertex) == 5, "неверное число вершин в списке _vertex класса LinkedGraph"
map2.add_link(Link(v2, v1))
assert len(map2._links) == 5, "метод add_link() добавил связь Link(v2, v1), хотя уже имеется связь Link(v1, v2)"
path = map2.find_path(v1, v5)
s = sum([x.dist for x in path[1]])
assert s == 3, "неверная суммарная длина маршрута, возможно, некорректно работает объект-свойство dist"
assert issubclass(Station, Vertex) and issubclass(LinkMetro, Link), ("класс Station должен наследоваться от класса Vertex,"
                                                                     " а класс LinkMetro от класса Link")

map2 = LinkedGraph()
v1 = Station("1")
v2 = Station("2")
v3 = Station("3")
v4 = Station("4")
v5 = Station("5")

map2.add_link(LinkMetro(v1, v2, 1))
map2.add_link(LinkMetro(v2, v3, 2))
map2.add_link(LinkMetro(v2, v4, 7))
map2.add_link(LinkMetro(v3, v4, 3))
map2.add_link(LinkMetro(v4, v5, 1))

assert len(map2._links) == 5, "неверное число связей в списке _links класса LinkedGraph"
assert len(map2._vertex) == 5, "неверное число вершин в списке _vertex класса LinkedGraph"
path = map2.find_path(v1, v5)
assert str(path[0]) == '[1, 2, 3, 4, 5]', path[0]
s = sum([x.dist for x in path[1]])
assert s == 7, "неверная суммарная длина маршрута для карты метро"


print('Все тесты пройдены')