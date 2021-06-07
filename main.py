import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Сигмоидная функция активации
    :param z: сумма скалярного произведения весов и входов плюс смещение
    :return: прогноз
    """
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, W, b):
    """
    Вычисляет операцию прямого распредаления перцептрона и возвращяет результат результат
    после применения сигмовидной функции активации
    :param X: входные данные
    :param W: веса
    :param b: смещение
    :return: прогноз
    """
    weighted_sum = np.dot(X, W) + b
    prediction = sigmoid(weighted_sum)
    return prediction

def calculate_error(y, y_predicted):
    """
    Вычисляет ошибку бинарной кросс-ентропии
    :param y: целевые метки
    :param y_predicted: прогнозируемые метки
    :return: ошибка
    """
    loss = np.sum(- y * np.log(y_predicted) - (1 - y) * np.log(1 - y_predicted))
    return loss

def gradient(X, Y, Y_predicted):
    """
    Градиент весов и смещения
    :param X: Входные данные
    :param Y: целевые метки
    :param Y_predicted: прогнозируемые метки
    :return: Производная ошибки по весам, производная ошибки по смещению
    """
    Error = Y_predicted - Y
    dW = np.dot(X.T, Error) # Вычисляем производную ошибки по весу т.е. (target - output) * x
    db = np.sum(Error) # Вычисляем производную ошибки по смещению
    return dW, db

def update_parameters(W, b, dW, db, learning_rate):
    """
    обновляем значения весов и смещения
    :param W: вес
    :param b: смещение
    :param dW: производную ошибки по весу
    :param db: производную ошибки по смещению
    :param learning_rate: скорость обучения
    :return: Вес, смещение
    """
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def train(X, Y, learning_rate, W, b, epochs, losses):
    """
    Обучаем перцептрон с помощью пакетного обновления
    :param X: входные данные
    :param Y: целевые метки
    :param learning_rate: скорость обучения
    :param W: веса
    :param b: смещения
    :param epochs: эпохи
    :param losses: потери
    :return: веса, смещения, потери
    """
    for i in range(epochs):
        Y_predicted = forward_propagation(X, W, b)
        losses[i, 0] = calculate_error(Y, Y_predicted)
        dW, db = gradient(X, Y, Y_predicted)
        W, b = update_parameters(W, b, dW, db, learning_rate)
    return W, b, losses

# Initializing parameters
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # declaring two data points
Y = np.array([0, 0, 0, 1]) # actual label
weights = np.array([0.0, 0.0]) # weights of perceptron
bias = 0.0 # bias value
epochs = 10000 # total epochs
learning_rate = 0.01 # learning rate
losses = np.zeros((epochs, 1)) # compute loss
print("Before training")
print("weights:", weights, "bias:", bias)
print("Target labels:", Y)

weights, bias, losses = train(X, Y, learning_rate, weights, bias, epochs, losses)

# Evaluating the performance
plt.figure()
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()
plt.savefig('output/legend.png')

print("\nAfter training")
print("weights:", weights, "bias:", bias)

# Predict value
A2 = forward_propagation(X, weights, bias)
pred = (A2 > 0.5) * 1
print("Predicted labels:", pred)

