import idx2numpy
import matplotlib.pyplot as plt

# Пути к файлам
TRAIN_IMAGE_FILENAME = '../data/mnist/train-images.idx3-ubyte'
TRAIN_LABEL_FILENAME = '../data/mnist/train-labels.idx1-ubyte'

# Загрузка данных
train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)

# Функция для отображения цифры
def show_digit(image, index):
    plt.imshow(image, cmap='gray')
    plt.title(f"Вариант {index + 1}")
    plt.axis('off')
    plt.show()

# Ваш вариант — цифра 4
digit = 4

# Получаем индексы всех изображений с этой цифрой
indices = (train_labels == digit).nonzero()[0]

# Выводим первые 5 вариантов
for i in range(5):
    show_digit(train_images[indices[i]], i)