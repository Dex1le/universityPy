def main():
    a = input()
    st = int(input())

    if st == 1:
        print(a)
    else:
        counter = 0
        steps = 1

        st_list = [''] * st

        for a1 in a:
            st_list[counter] += a1

            if counter+1 == st:
                steps = -1
            elif counter == 0:
                steps = 1

            counter += steps

        print(''.join(st_list))

if __name__ == '__main__':
    main()