def main():

    a = int(input())
    pol = 0
    n_a = a
    pr = a

    def ch(pr):
        k = 0
        while pr > 0:
            pr = pr // 10
            k += 1
        return k

    if a >= 0:
        while n_a > 0:
            n = n_a % 10
            pol += 10 ** (ch(n_a) - 1) * n
            n_a = (n_a // 10)

    else:
        pr = a * (-1)
        n_a = a * (-1)
        while n_a > 0:
            n = n_a % 10
            pol += 10 ** (ch(n_a) - 1) * n
            n_a = (n_a // 10)
        pol = pol * (-1)

    if -128 <= pol <= 128:
        print(pol)

    else:
        print('no solution')

if __name__ == '__main__':
    main()