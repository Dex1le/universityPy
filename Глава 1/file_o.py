def main():

    a = int(input())
    pol = 0
    a1 = a
    a2 = a

    def ch(a2):
        k = 0

        while a2 > 0:
            a2 = a2 // 10
            k += 1
        return k

    while a1 > 0:
        n = a1 % 10
        pol += 10 ** (ch(a1) - 1) * n
        a1 = (a1 // 10)
        print(pol, a1)

    if pol == a:
        print('True')

    else:
        print('False')

if __name__ == '__main__':
    main()