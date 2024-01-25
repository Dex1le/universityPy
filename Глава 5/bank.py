def rob_banks():
    pora_grabit = int(input("Введите количество банков на улице: "))
    banks = []
    for i in range(pora_grabit):
        name_bank = input("Введите название банка: ")
        money = int(input("Введите сумму денег в сейфе: "))
        banks.append((name_bank, money))
    max_money = [0] * pora_grabit
    list_banks = [0] * pora_grabit
    for i in range(pora_grabit):
        if i == 0:
            max_money[i] = banks[i][1]
        elif i == 1:
            max_money[i] = max(banks[i][1], banks[i-1][1])
            if max_money[i] == banks[i][1]:
                list_banks[i] = i
            else:
                list_banks[i] = i-1
        else:
            include = banks[i][1] + max_money[i-2]
            exclude = max_money[i-1]
            if include > exclude:
                max_money[i] = include
                list_banks[i] = i
            else:
                max_money[i] = exclude
                list_banks[i] = list_banks[i-1]
    print("Максимальная сумма денег:", max_money[-1], "млн рублей")

    return

if __name__ == "__main__":
    rob_banks()