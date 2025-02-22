# Neuro code for my victory:
# ============================
# 1) Импорт библиотек
# ============================

import os
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ============================
# 2) Настройка GPU
# ============================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs настроены для динамического роста памяти.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU не обнаружены, будет использоваться CPU.")

# ============================
# 3) Настройка многопоточности (параллелизм)
# ============================
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
print("Параллелизм настроен: 4 потока для intra-op и inter-op.")

# ============================
# 4) Определяем параметры модели.
# ============================
# Параметры, которые можно изменять
IMG_WIDTH, IMG_HEIGHT = 244, 244       # Размер изображений (входной слой)
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)  # 3-канальное изображение (RGB)
NUM_CLASSES = len(os.listdir(r"C:\Users\Егор Васильев\Downloads\lfw-deepfunneled")) # Число классов (выходной слой)

# Гиперпараметры
LEARNING_RATE = 0.001                # Параметр, отвечающий за изменение весов
EPOCHS = 2                         # Количество эпох обучения (можно менять)
BATCH_SIZE = 16

# Коэффициент L2 регуляризации (можно поэкспериментировать с этим значением)
L2_REG = 0.001

# Параметры архитектуры модели:
# Список параметров для сверточных слоев: (фильтры, размер ядра)
conv_layers_params = [
    (32, (3, 3)),
    (64, (3, 3))
]
# Параметры полносвязных (Dense) слоев (количество нейронов)
dense_layers_units = [128]

def build_model(input_shape, num_classes, conv_params, dense_units, learning_rate):
    """
    Создает и компилирует модель на основе заданных параметров.
    """
    model = Sequential()

    # Добавляем сверточные слои с max pooling
    for idx, (filters, kernel_size) in enumerate(conv_params):
        if idx == 0:
            model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=input_shape,
                             kernel_regularizer=l2(L2_REG)))
        else:
            model.add(Conv2D(filters, kernel_size, activation='relu',
                             kernel_regularizer=l2(L2_REG)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Дополнительно можно добавить Dropout между сверточными блоками (например, 0.25)
        model.add(Dropout(0.25))

    # Вместо Flatten() используем глобальный pooling для снижения размерности
    model.add(GlobalAveragePooling2D())

    # Полносвязные слои
    for units in dense_units:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(L2_REG)))
        model.add(Dropout(0.5))

    # Выходной слой
    model.add(Dense(num_classes, activation='softmax'))

    # Компиляция модели
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ============================
# 5) Загрузка данных и обучение
# ============================
# Запрашиваем у пользователя пути к папкам обучения и тестирования
train_folder = r"C:\Users\Егор Васильев\Downloads\lfw-deepfunneled"
test_folder = r"C:\Users\Егор Васильев\Downloads\lfw-deepfunneled"

# Создаем генераторы данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.1  # 10% данных для валидации
)

# Для тестирования данные только масштабируются
test_datagen = ImageDataGenerator(rescale=1./255)

# Генератор для обучения
train_generator = train_datagen.flow_from_directory(
    directory=train_folder,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Генератор для валидации
validation_generator = train_datagen.flow_from_directory(
    directory=train_folder,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Генератор для тестирования
test_generator = test_datagen.flow_from_directory(
    directory=test_folder,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Создаем модель
model = build_model(INPUT_SHAPE, NUM_CLASSES, conv_layers_params, dense_layers_units, LEARNING_RATE)
model.summary()

# Колбэки: ReduceLROnPlateau - уменьшаем learning rate при отсутствии улучшения
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
# Если модель слишком рано останавливается, можно не использовать EarlyStopping или увеличить patience
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Обучаем модель с валидацией
print("\nНачало обучения...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[lr_reduction],
    verbose=1
)

# ============================
# 6) Тестирование модели
# ============================
print("\nНачало тестирования модели на тестовом наборе...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nТочность на тестовом наборе: {test_accuracy*100:.2f}%")

# ============================
# 7) Сохранение модели по выбору пользователя
# ============================
save_choice = input("\nСохранить обученную модель? (y/n): ").strip().lower()
if save_choice == 'y':
    now = datetime.datetime.now()
    # Корректно оформляем путь (используем raw-строку)
    folder_path = r"D:\нейронки ебутся"
    filename = now.strftime("%H.%M-%d.%m.%Y.h5")
    full_path = os.path.join(folder_path, filename)
    model.save(full_path)
    print(f"Модель сохранена в файле: {full_path}")
else:
    print("Модель не сохранена.")

# Конец программы.
