def find_expression(numbers, target_sum, current_sum, expression, first_num):
    if not numbers:
        if current_sum == target_sum:
            expression = f"{first_num}{expression}={target_sum}"
            print(expression)
            exit()
        else:
            return

    number = numbers[0]

    # Вызываем функции для сложения и вычитания
    find_expression(numbers[1:], target_sum, current_sum + number, expression + "+" + str(number), first_num)
    find_expression(numbers[1:], target_sum, current_sum - number, expression + "-" + str(number), first_num)

# Считываем входные данные из файла
with open("/Users/dex1le/PycharmProjects/university/Lab1/input.txt") as file:
    input_data = file.readline().split()
    n = int(input_data[0])
    numbers = [int(input_data[i]) for i in range(1, n + 1)]
    target_sum = int(input_data[-1])

    first_num = numbers.pop(0)

# Запускаем функцию для поиска решения
find_expression(numbers, target_sum, first_num, "", first_num)