def spiral_order(matrix):
    result = []
    if not matrix:
        return result

    row_begin = 0
    row_end = len(matrix) - 1
    col_begin = 0
    col_end = len(matrix[0]) - 1

    while row_begin <= row_end and col_begin <= col_end:
        # Верхняя строка
        for i in range(col_begin, col_end + 1):
            result.append(matrix[row_begin][i])
        row_begin += 1

        # Правый столбец
        for i in range(row_begin, row_end + 1):
            result.append(matrix[i][col_end])
        col_end -= 1

        # Нижняя строка
        if row_begin <= row_end:
            for i in range(col_end, col_begin - 1, -1):
                result.append(matrix[row_end][i])
            row_end -= 1

        # Левый столбец
        if col_begin <= col_end:
            for i in range(row_end, row_begin - 1, -1):
                result.append(matrix[i][col_begin])
            col_begin += 1

    return result

def main():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix1 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    print(spiral_order(matrix))
    print(spiral_order(matrix1))

if __name__ == "__main__":
    main()