def rotate(matrix):
    # Получаем размеры матрицы
    rows = len(matrix)
    cols = len(matrix[0])

    # Создаем новую матрицу с повернутыми значениями
    rotated_matrix = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            # Поворачиваем элементы матрицы
            rotated_matrix[j][rows - i - 1] = matrix[i][j]

    return rotated_matrix

def main():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(rotate(matrix))

if __name__ == "__main__":
    main()