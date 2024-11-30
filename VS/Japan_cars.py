import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.model_selection import train_test_split

from keras import models
from keras import layers
from keras.utils import to_categorical

dataset = np.loadtxt('https://storage.yandexcloud.net/academy.ai/A_Z_Handwritten_Data.csv', delimiter=',')
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

X = dataset[:,1:785]
Y = dataset[:,0]

(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.2, shuffle=True)

for i in range(40):
    x = x_train[i]
    x = x.reshape((28, 28))
    plt.axis('off')
    im = plt.subplot(5, 8, i+1)
    plt.title(word_dict.get(y_train[i]))
    im.imshow(x, cmap='gray')

model = models.Sequential()
model.add(layers.Dense(8192, activation='relu', input_shape=(784,)))
model.add(layers.Dense(4096, activation='relu', input_shape=(784,)))
#model.add(layers.Dense(2048, activation='relu', input_shape=(784,)))
#model.add(layers.Dense(1024, activation='relu', input_shape=(784,)))
#model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
model.add(layers.Dense(26, activation='softmax'))

model.compile(optimizer='rmsprop',
 loss='categorical_crossentropy',
 metrics=['accuracy'])

# Задаем тип данным и нормируем на максимальное значение в тензоре (приводим к диапазону [0, 1])
train_images = x_train.astype('float32') / 255

# Задаем тип данным и нормируем на максимальное значение в тензоре (приводим к диапазону [0, 1])
test_images = x_test.astype('float32') / 255

train_label = to_categorical(y_train, 26) # Кодируем обучающие метки на 26 классов
test_label = to_categorical(y_test, 26)   # Кодируем тестовые метки на 26 классов

history = model.fit(train_images, train_label, validation_data=(test_images, test_label), epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_label)
print('Точность на тестовом образце:', test_acc)
print('Потери на тестовом образце:', test_loss)

# Выбор нужной картинки из тестовой выборки
n = 24
x = test_images[n]

# Массив из одного примера, так как нейронка принимает именно массивы примеров (батчи) для распознавания
x = np.expand_dims(x, axis=0)

# Проверка формы данных
print(x.shape)

# Предсказываем выбранную картинку
prediction = model.predict(x)

# Вывод результата - вектор из 10 чисел
print(f'Вектор результата на 10 выходных нейронах:\n{prediction}')

# Получение и вывод индекса самого большого элемента (это значение цифры, которую распознала сеть)
pred = np.argmax(prediction)

print(f'Распознана цифра: {pred}, соответствует символу "{word_dict[pred]}"')
print(f'Правильное значение: {np.argmax(test_label[n])}')

digit = test_images[n]
digit = digit.reshape((28, 28))
fig, ax = plt.subplots(1,1)
ax.set_title(f'Образец с индеком {n} из нашего набора данных\nраспознан как цифра {pred}, что соответствует букве "{word_dict[pred]}" в словаре')
ax.imshow(digit, cmap=plt.cm.binary)
plt.show()

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Потери на обучающей выборке')
plt.plot(epochs, val_loss_values, 'b', label='Потери на тестовой выборке')
plt.title('График потерь')
plt.xlabel('Эпоха обучения')
plt.ylabel('Потери')
plt.legend()
plt.show()


