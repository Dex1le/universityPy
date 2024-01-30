def main(a,b):
    l1 = list()
    l2 = list()

    l3 = a.copy()
    l4 = b.copy()

    x = y = z = w = 0

    for i in a:
        if i in b:
            l1.append(i)
            l4.remove(i)
        else:
            l2.append(i)
    for i in b:
        if i not in a:
            l2.append(i)
        if i in a:
            l3.remove(i)

    x = len(l1)
    y = len(l2)
    z = len(l3)
    w = len(l4)

    return l1,l2,l3,l4

if  __name__ =="__main__":
    a = [0, 33, 37, 6, 10, 44, 13, 47, 16, 18, 22, 25]
    b = [1, 38, 48, 8, 41, 7, 12, 47, 16, 40, 20, 23, 25]
    print(main(a,b))
