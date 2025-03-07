import numpy as np
import matplotlib.pyplot as plt

# Активационные функции и их производные
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Обучение многослойного персептрона
def train_mlp(X, y, hidden_neurons=8, cycles=20000, learning_rate=0.1, error_threshold=0.001):
    input_neurons = X.shape[1]
    output_neurons = 1

    # Случайная инициализация весов
    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

    bias_hidden = np.random.uniform(size=(1, hidden_neurons))
    bias_output = np.random.uniform(size=(1, output_neurons))

    errors = []

    for cycle in range(cycles):
        # Прямой проход (forward propagation)
        hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_activation)

        # Ошибка
        error = y - predicted_output
        mean_error = np.mean(np.abs(error))
        errors.append(mean_error)

        # Проверка условия завершения
        if mean_error < error_threshold:
            print(f"Обучение завершено на цикле {cycle + 1} с ошибкой {mean_error}")
            break

        # Обратное распространение ошибки (backpropagation)
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Обновление весов
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
        bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # График ошибки
    plt.plot(errors, label='Ошибка обучения')
    plt.title('Ошибка на протяжении всех циклов обучения')
    plt.xlabel('Циклы')
    plt.ylabel('Средняя абсолютная ошибка')
    plt.grid(True)
    plt.legend()
    plt.show()

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, errors

# Проверка работы многослойного персептрона
def test_mlp(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_activation = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)

    return predicted_output

# Входные данные (таблица истинности для четырех входов)
X = np.array([[0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 1, 1],
              [0, 1, 0, 0],
              [0, 1, 0, 1],
              [0, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 0, 1],
              [1, 0, 1, 0],
              [1, 0, 1, 1],
              [1, 1, 0, 0],
              [1, 1, 0, 1],
              [1, 1, 1, 0],
              [1, 1, 1, 1]])

# Ожидаемые выходы (например, XOR для 4 переменных; замени на нужные значения)
y = np.array([[0], [1], [1], [0], [1], [0], [0], [1], [1], [0], [0], [1], [0], [0], [1], [1]])

# Обучение многослойного персептрона
weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, errors = train_mlp(X, y)

# Тестирование
predictions = test_mlp(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

# Округляем предсказанные значения до ближайшего 0 или 1
rounded_predictions = np.round(predictions)

# Вывод таблицы истинности
print("Таблица истинности (после обучения):")
print("x1 | x2  | x3  | x4  | Ожидалось |  Предсказано")
for i in range(len(X)):
    print(f"{X[i][0]}  |  {X[i][1]}  |  {X[i][2]}  |  {X[i][3]}  |     {y[i][0]}     |     {int(rounded_predictions[i][0])}")


# Итоговый график ошибки
plt.plot(errors, label='Итоговая ошибка')
plt.title('Ошибка на протяжении всех циклов обучения')
plt.xlabel('Циклы')
plt.ylabel('Средняя абсолютная ошибка')
plt.grid(True)
plt.legend()
plt.show()