class BaseClass:
    num_base_calls = 0

    def call_me(self) -> None:
        print("Calling method on BaseClass")
        self.num_base_calls += 1


class LeftSubClass(BaseClass):
    num_left_calls = 0

    def call_me(self) -> None:
        super().call_me()
        print('Calling method on LeftSubClass')
        self.num_left_calls += 1


class RightSubClass(BaseClass):
    num_right_calls = 0

    def call_me(self) -> None:
        super().call_me()
        print('Calling method on RightSubClass')
        self.num_right_calls += 1


class SubClass(LeftSubClass, RightSubClass):
    num_sub_calls = 0

    def call_me(self) -> None:
        super().call_me()
        print('Calling method on SubClass')
        self.num_sub_calls += 1


s = SubClass()
s.call_me()
print(SubClass.mro())

print(s.num_base_calls)
print(s.num_left_calls)
print(s.num_right_calls)
print(s.num_sub_calls)