import numpy as np
import matplotlib.pyplot as plt

# Активационные функции и их производные
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Обучение многослойного персептрона
def train_mlp(X, y, hidden_neurons_1=8, hidden_neurons_2=6, cycles=20000, learning_rate=0.1, error_threshold=0.001):
    input_neurons = X.shape[1]
    output_neurons = 1

    # Случайная инициализация весов
    weights_input_hidden1 = np.random.uniform(size=(input_neurons, hidden_neurons_1))
    weights_hidden1_hidden2 = np.random.uniform(size=(hidden_neurons_1, hidden_neurons_2))
    weights_hidden2_output = np.random.uniform(size=(hidden_neurons_2, output_neurons))

    bias_hidden1 = np.random.uniform(size=(1, hidden_neurons_1))
    bias_hidden2 = np.random.uniform(size=(1, hidden_neurons_2))
    bias_output = np.random.uniform(size=(1, output_neurons))

    errors = []

    for cycle in range(cycles):
        # Прямой проход (forward propagation)
        hidden_layer1_activation = np.dot(X, weights_input_hidden1) + bias_hidden1
        hidden_layer1_output = sigmoid(hidden_layer1_activation)

        hidden_layer2_activation = np.dot(hidden_layer1_output, weights_hidden1_hidden2) + bias_hidden2
        hidden_layer2_output = sigmoid(hidden_layer2_activation)

        output_layer_activation = np.dot(hidden_layer2_output, weights_hidden2_output) + bias_output
        predicted_output = sigmoid(output_layer_activation)

        # Ошибка
        error = y - predicted_output
        mean_error = np.mean(np.abs(error))
        errors.append(mean_error)

        # Проверка условия завершения
        if mean_error < error_threshold:
            print(f"Обучение завершено на цикле {cycle + 1} с ошибкой {mean_error}")
            break

        # Обратное распространение ошибки
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden2 = d_predicted_output.dot(weights_hidden2_output.T)
        d_hidden2 = error_hidden2 * sigmoid_derivative(hidden_layer2_output)

        error_hidden1 = d_hidden2.dot(weights_hidden1_hidden2.T)
        d_hidden1 = error_hidden1 * sigmoid_derivative(hidden_layer1_output)

        # Обновление весов
        weights_hidden2_output += hidden_layer2_output.T.dot(d_predicted_output) * learning_rate
        weights_hidden1_hidden2 += hidden_layer1_output.T.dot(d_hidden2) * learning_rate
        weights_input_hidden1 += X.T.dot(d_hidden1) * learning_rate

        bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        bias_hidden2 += np.sum(d_hidden2, axis=0, keepdims=True) * learning_rate
        bias_hidden1 += np.sum(d_hidden1, axis=0, keepdims=True) * learning_rate

    # График ошибки
    plt.plot(errors, label='Ошибка обучения')
    plt.title('Ошибка на протяжении всех циклов обучения')
    plt.xlabel('Циклы')
    plt.ylabel('Средняя абсолютная ошибка')
    plt.grid(True)
    plt.legend()
    plt.show()

    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, bias_hidden1, bias_hidden2, bias_output, errors

# Проверка работы многослойного персептрона
def test_mlp(X, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, bias_hidden1, bias_hidden2, bias_output):
    hidden_layer1_activation = np.dot(X, weights_input_hidden1) + bias_hidden1
    hidden_layer1_output = sigmoid(hidden_layer1_activation)

    hidden_layer2_activation = np.dot(hidden_layer1_output, weights_hidden1_hidden2) + bias_hidden2
    hidden_layer2_output = sigmoid(hidden_layer2_activation)

    output_layer_activation = np.dot(hidden_layer2_output, weights_hidden2_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)

    return predicted_output

# Входные данные (таблица истинности для четырёх входов)
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
weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, bias_hidden1, bias_hidden2, bias_output, errors = train_mlp(X, y)

# Тестирование
predictions = test_mlp(X, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, bias_hidden1, bias_hidden2, bias_output)

# Округляем предсказанные значения до ближайшего 0 или 1
rounded_predictions = np.round(predictions)

# Вывод таблицы истинности
print("Таблица истинности (после обучения):")
print("x1 | x2  | x3  | x4  | Ожидалось |  Предсказано")
for i in range(len(X)):
    print(f"{X[i][0]}  |  {X[i][1]}  |  {X[i][2]}  |  {X[i][3]}  |     {y[i][0]}     |     {int(rounded_predictions[i][0])}")