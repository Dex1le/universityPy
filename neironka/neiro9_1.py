import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Загрузка изображений
china = load_sample_image("china.jpg") / 255.0
flower = load_sample_image("flower.jpg") / 255.0
images = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = images.shape

# Создание фильтров (7x7, 2 фильтра: вертикальный и горизонтальный)
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # вертикальный
filters[3, :, :, 1] = 1  # горизонтальный

# Прямая свёртка (фрагмент 2)
outputs_manual = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

# Conv2D слой с такими же весами (фрагмент 3)
conv_layer = tf.keras.layers.Conv2D(filters=2, kernel_size=7, strides=1,
                                    padding="SAME", activation=None,
                                    input_shape=(None, None, 3))
conv_layer.build(images.shape)
conv_layer.set_weights([filters, np.zeros(2)])  # веса и bias

# Получаем выход из Conv2D слоя
outputs_layer = conv_layer(images)

# Функция отрисовки
def plot_output(title, tensor):
    plt.figure(figsize=(8, 4))
    for img_idx in range(2):
        for fmap_idx in range(2):
            plt.subplot(2, 2, img_idx * 2 + fmap_idx + 1)
            plt.imshow(tensor[img_idx, :, :, fmap_idx], cmap='gray')
            plt.axis('off')
            plt.title(f'{title}: Image {img_idx+1}, Filter {fmap_idx+1}')
    plt.tight_layout()
    plt.show()

# Показываем результаты
plot_output("Manual conv2d", outputs_manual.numpy())
plot_output("Keras Conv2D", outputs_layer.numpy())
