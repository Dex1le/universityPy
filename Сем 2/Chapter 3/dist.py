from math import *
from abc import ABC, abstractmethod


class Sample:
    def __init__(self, sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

        def __repr__(self) -> str:
            return (
                f"{self.__class__.__name__}("
                f"sepal_length={self.sepal_length}, "
                f"sepal_width={self.sepal_width}, "
                f"petal_length={self.petal_length}, "
                f"petal_width={self.petal_width}, "
            )


class Distance(ABC):
    """Определение расстояния между двумя точками"""

    @abstractmethod
    def distance(self, s1: 'Sample', s2: 'Sample') -> float:
        pass


class ED(Distance):
    def distance(self, s1: 'Sample', s2: 'Sample') -> float:
        return hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width
        )


class MD(Distance):
    def distance(self, s1: 'Sample', s2: 'Sample'):
        return sum([
            abs(s1.sepal_length - s2.sepal_length),
            abs(s1.sepal_width - s2.sepal_width),
            abs(s1.petal_length - s2.petal_length),
            abs(s1.petal_width - s2.petal_width)
            ]
        )


class CD(Distance):
    def distance(self, s1: 'Sample', s2: 'Sample'):
        return max([
            abs(s1.sepal_length - s2.sepal_length),
            abs(s1.sepal_width - s2.sepal_width),
            abs(s1.petal_length - s2.petal_length),
            abs(s1.petal_width - s2.petal_width)
            ]
        )


s1 = Sample(1, 2, 3, 4)
s2 = Sample(8, 5, 7, 4)

ed = ED()
md = MD()
cd = CD()

print(ed.distance(s1, s2))
print(md.distance(s1, s2))
print(cd.distance(s1, s2))