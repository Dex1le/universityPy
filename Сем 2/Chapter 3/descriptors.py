from dataclasses import dataclass, field, InitVar

@dataclass
class V3D:
    x: int = field(repr=False)
    y: int
    z: int = field(compare=False)
    length: float = field(init=False, compare=False, default=0)
    calc_len: InitVar[bool] = True

    def __post_init__(self, calc_len: bool):
        if calc_len:
            self.length = (self.x * self.x + self.y * self.y + self.z * self.z) ** 0.5

v = V3D(0, 2, 3)
print(v)
#print (v.__dict__)

v = V3D(0, 2, 3)
v2 = V3D(0, 2, 3)
print(v)
print(v == v2)
print()





