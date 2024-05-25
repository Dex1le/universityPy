import collections
from math import *
from abc import ABC, abstractmethod
from typing import Optional, Union, Iterable, Counter
import weakref
import datetime


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


class KnownSample(Sample):
    def __init__(
            self,
            species: str,
            sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float
    ) -> None:
        super().__init__(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )
        self.species = species

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"species={self.species!r}, "
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f")"
        )


class TrainingKnownSample(KnownSample):
    pass


class TestingKnownSample(KnownSample):
    """Данные тестирования"""

    def __init__(
            self,
            species: str,
            sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float,
            classification: Optional[str] = None
    ) -> None:

        super().__init__(
            species=species,
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )
        self.classification = classification

    def matches(self) -> bool:
        return self.species == self.classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"species={self.species!r}, "
            f"classification={self.classification!r}, "
            f")"
        )


class UnkownSample(Sample):
    pass


class ClassifiedSample(Sample):
    def __init__(self, classification: str, sample: UnkownSample) -> None:
        super().__init__(
            sepal_length=sample.sepal_length,
            sepal_width=sample.sepal_width,
            petal_length=sample.petal_length,
            petal_width=sample.petal_width
        )
        self.classification = classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"classification={self.classification!r}, "
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f")"
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


class Hyperparameter:
    def __init__(self, k: int, alg: Distance, training:'TrainingData'):
        self.k = k
        self.alg = alg
        self.data:weakref.ReferenceType['TrainingData'] = weakref.ref(training)

        self.quality:Optional[float] = None

    def classify(self, sample: Union[UnkownSample, TestingKnownSample]) -> str:
        training_data: "TrainingData" = self.data()
        if not training_data:
            raise RuntimeError("No Training Data")

        distance: list[tuple[float, TrainingData]] = sorted(((self.alg.distance(sample, known), known) for known in training_data.training), key=lambda x: x[0])
        k_nearest: tuple[str] = (known.species for d,known in distance[:self.k])

        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *other = frequency.most_common()
        species,_ = best_fit
        return species

    def test(self) -> None:
        training_data: "TrainingData" = self.data()
        if not training_data:
            raise RuntimeError("No Training Data")

        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classification = self.classify(sample)
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1

        self.quality = pass_count / (pass_count + fail_count)


class TrainingData:
    def __init__(self, name: str) -> None:
        self.name = name

        self.training: list[TrainingKnownSample] = []
        self.testing: list[TestingKnownSample] = []
        self.uploaded: datetime.datetime = None

        self.tuning: list[Hyperparameter] = []
        self.tuningTime: datetime.datetime = None

    def loaded(self, raw_data_iter: Iterable[dict[str, str]]) -> None:
        for n, row in enumerate(raw_data_iter):
            if n % 5 == 0:
                test = TestingKnownSample(spicies=row['spicies'],
                                          sepal_length=row['sepal_length'],
                                          sepal_width=row['sepal_width'],
                                          petal_length=row['petal_length'],
                                          petal_width=row['petal_width'])
                self.testing(test)
            else:
                training = TrainingKnownSample(spicies=row['spicies'],
                                               sepal_length=row['sepal_length'],
                                               sepal_width=row['sepal_width'],
                                               petal_length=row['petal_length'],
                                               petal_width=row['petal_width']
                                               )
                self.training(training)
            self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parametr: Hyperparameter) -> None:
        parametr.data = weakref.ref(self)
        parametr.test()
        self.tuning.append(parametr)
        self.tuningTime = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parametr: Hyperparameter, sample: UnkownSample) -> ClassifiedSample:
        parametr.data = weakref.ref(self)
        return ClassifiedSample(parametr.classify(sample), sample)


td = TrainingData("test1")
s1 = TestingKnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-S")
td.testing.append(s1)

t1 = TrainingKnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species="Iris-S")
t2 = TrainingKnownSample(sepal_length=7.9, sepal_width=3.2, petal_length=4.7, petal_width=1.4, species="Iris-V")
td.training.extend([t1, t2])

h = Hyperparameter(k=3, alg=MD(), training=td)

u = UnkownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2)

h.classify(u)

td.test(h)
print(td.classify(h, u))

print(td.tuning[0].quality)
# s1 = Sample(3, 2, 4, 3)
# s2 = Sample(8, 5, 7, 4)
#
# ed = ED()
# md = MD()
# cd = CD()
#
# print(ed.distance(s1, s2))
# print(md.distance(s1, s2))
# print(cd.distance(s1, s2)