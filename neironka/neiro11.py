import os  # Работа с файловой системой
import numpy as np  # Научные вычисления, массивы
import pretty_midi  # Работа с MIDI-файлами
import tensorflow as tf  # Основной фреймворк для нейросетей
from tensorflow.keras.models import Model, load_model  # Импорт для построения и загрузки моделей
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout  # Слои для нейросети
from tensorflow.keras.utils import to_categorical  # Для one-hot кодирования

# === Параметры
MIDI_DIR = "messiah"  # Папка с обучающими MIDI-файлами
MODEL_PATH = "einaudi_model_clean.h5"  # Путь до модели
GENERATED_MIDI = "handel_generated_clean.mid"  # Куда сохранить сгенерированный MIDI
SEQ_LENGTH = 50  # Длина последовательности для обучения
EPOCHS = 30  # Количество эпох обучения
TEMPERATURE = 0.8  # Температура для сэмплирования
GENERATE_NOTES = 300  # Сколько нот сгенерировать

# === Извлечение нот (pitch + duration)
def extract_notes(midi_path):  # Функция для извлечения нот из MIDI
    midi = pretty_midi.PrettyMIDI(midi_path)  # Загружаем MIDI-файл
    notes = []  # Список нот
    for inst in midi.instruments:  # Перебираем инструменты
        if not inst.is_drum:  # Игнорируем ударные
            for note in inst.notes:  # Перебираем ноты
                dur = round(note.end - note.start, 2)  # Вычисляем длительность
                notes.append((note.pitch, dur))  # Сохраняем питч и длительность
    return notes  # Возвращаем список нот

# === Загрузка данных
all_notes = []  # Сюда соберём все ноты со всех файлов
for fname in os.listdir(MIDI_DIR):  # Перебираем файлы в папке
    if fname.endswith(".mid"):  # Только .mid файлы
        try:
            all_notes.extend(extract_notes(os.path.join(MIDI_DIR, fname)))  # Добавляем ноты из файла
        except Exception as e:  # Обработка ошибок
            print(f"Ошибка в {fname}: {e}")  # Вывод ошибки

if len(all_notes) < SEQ_LENGTH + 1:  # Проверка, достаточно ли данных
    raise ValueError("Недостаточно данных для обучения")  # Ошибка если мало нот

# Разделяем ноты на питчи и длительности
pitches = [n[0] for n in all_notes]
durations = [n[1] for n in all_notes]
unique_pitches = sorted(set(pitches))  # Уникальные питчи
unique_durations = sorted(set(durations))  # Уникальные длительности

# Создание словарей для кодирования и декодирования
pitch2idx = {p: i for i, p in enumerate(unique_pitches)}
dur2idx = {d: i for i, d in enumerate(unique_durations)}
idx2pitch = {i: p for p, i in pitch2idx.items()}
idx2dur = {i: d for d, i in dur2idx.items()}

# Кодируем ноты в индексы
encoded = [(pitch2idx[p], dur2idx[d]) for p, d in all_notes]

# === Подготовка выборок для обучения
X, y_pitch, y_dur = [], [], []  # X — входы, y — целевые выходы
for i in range(len(encoded) - SEQ_LENGTH):  # Идём по всем последовательностям длины SEQ_LENGTH
    X.append([e[0] for e in encoded[i:i+SEQ_LENGTH]])  # Вход: питчи
    y_pitch.append(encoded[i+SEQ_LENGTH][0])  # Следующий питч
    y_dur.append(encoded[i+SEQ_LENGTH][1])  # Следующая длительность

# Преобразуем в numpy-массивы
X = np.array(X)
y_pitch = to_categorical(y_pitch, num_classes=len(unique_pitches))  # One-hot кодируем питчи
y_dur = to_categorical(y_dur, num_classes=len(unique_durations))  # One-hot кодируем длительности

# === Модель: загрузка или создание
if os.path.exists(MODEL_PATH):  # Если модель уже есть
    print("Загружаем модель...")  # Выводим сообщение
    model = load_model(MODEL_PATH)  # Загружаем модель
else:
    print("Создаём новую модель...")  # Создание новой модели
    inp = Input(shape=(SEQ_LENGTH,))  # Вход: последовательность питчей
    x = Embedding(input_dim=len(unique_pitches), output_dim=100)(inp)  # Эмбеддинг для питчей
    x = LSTM(128)(x)  # LSTM слой
    x = Dropout(0.2)(x)  # Dropout для регуляризации
    out_pitch = Dense(len(unique_pitches), activation="softmax", name="pitch")(x)  # Выход: предсказание питча
    out_dur = Dense(len(unique_durations), activation="softmax", name="duration")(x)  # Выход: предсказание длительности
    model = Model(inputs=inp, outputs=[out_pitch, out_dur])  # Собираем модель
    model.compile(  # Компилируем
        loss={"pitch": "categorical_crossentropy", "duration": "categorical_crossentropy"},
        optimizer="adam"
    )

# === Обучение модели
model.fit(X, {"pitch": y_pitch, "duration": y_dur}, batch_size=64, epochs=EPOCHS)  # Обучаем модель
model.save(MODEL_PATH)  # Сохраняем модель
print(f"Модель сохранена: {MODEL_PATH}")  # Сообщаем о сохранении

# === Генерация последовательности
def sample(preds, temperature=1.0):  # Функция сэмплирования с температурой
    preds = np.log(preds + 1e-8) / temperature  # Применяем температуру
    exp_preds = np.exp(preds)  # Экспоненцируем
    return np.random.choice(len(preds), p=exp_preds / np.sum(exp_preds))  # Выбираем индекс

# Начальное зерно для генерации
seed = [e[0] for e in encoded[:SEQ_LENGTH]]  # Начальная последовательность питчей
generated_pitches = seed[:]  # Копируем в массив генерации
generated_durations = []  # Пустой массив длительностей

# Генерация нот
for _ in range(GENERATE_NOTES):  # Генерируем указанное число нот
    input_seq = np.array(generated_pitches[-SEQ_LENGTH:]).reshape(1, -1)  # Последние SEQ_LENGTH нот
    pred_pitch, pred_dur = model.predict(input_seq, verbose=0)  # Предсказания
    next_pitch = sample(pred_pitch[0], TEMPERATURE)  # Сэмплируем следующий питч
    next_dur = sample(pred_dur[0], TEMPERATURE)  # Сэмплируем длительность
    generated_pitches.append(next_pitch)  # Добавляем в массив
    generated_durations.append(next_dur)  # То же с длительностью

# === Сбор в MIDI и сохранение
midi = pretty_midi.PrettyMIDI()  # Новый MIDI-объект
inst = pretty_midi.Instrument(program=0)  # Инструмент (акустическое пианино)
start = 0.0  # Начальное время

# Преобразуем индексы в ноты и добавляем в трек
for pitch_idx, dur_idx in zip(generated_pitches[SEQ_LENGTH:], generated_durations):  # Пропускаем seed
    pitch = idx2pitch[pitch_idx]  # Получаем питч
    dur = idx2dur[dur_idx]  # Получаем длительность
    velocity = np.random.randint(85, 110)  # Случайная громкость

    note = pretty_midi.Note(  # Создаём ноту
        velocity=velocity,
        pitch=pitch,
        start=start,
        end=start + dur
    )
    inst.notes.append(note)  # Добавляем ноту в инструмент
    start += dur  # Обновляем старт времени

midi.instruments.append(inst)  # Добавляем инструмент в MIDI-файл
midi.write(GENERATED_MIDI)  # Сохраняем результат
print(f"Сгенерировано: {GENERATED_MIDI}")  # Сообщение о завершении
