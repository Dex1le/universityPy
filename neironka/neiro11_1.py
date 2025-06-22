import numpy as np
import pretty_midi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

SEQ_LENGTH = 50
NUM_NOTES = 128
SAMPLES = 3000
EPOCHS = 50
BATCH_SIZE = 64

def generate_dense_melody():
    data = []
    for _ in range(SAMPLES):
        seq = []
        base_note = np.random.randint(48, 72)
        holding_note = False
        hold_length = 0
        for t in range(SEQ_LENGTH):
            if holding_note:
                note = base_note
                hold_length -= 1
                if hold_length == 0:
                    holding_note = False
            else:
                r = np.random.rand()
                if r < 0.1:
                    note = 0
                else:
                    step = np.random.choice([-2, -1, 0, 1, 2])
                    base_note = np.clip(base_note + step, 48, 72)
                    note = base_note
                    if np.random.rand() < 0.3:
                        holding_note = True
                        hold_length = np.random.randint(1, 4)
            seq.append(note)
        data.append(seq)
    return np.array(data, dtype=int)

print("Генерируем плотные мелодии...")
X = generate_dense_melody()

# y — следующая нота после последовательности
y = np.zeros((SAMPLES,), dtype=int)
for i in range(SAMPLES):
    y[i] = np.random.choice([0] + list(range(48, 73)))

def one_hot_encode_X(arr):
    result = np.zeros((arr.shape[0], arr.shape[1], NUM_NOTES))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            note = arr[i, j]
            result[i, j, note] = 1
    return result

X_oh = one_hot_encode_X(X)

y_oh = np.zeros((SAMPLES, NUM_NOTES))
for i in range(SAMPLES):
    y_oh[i, y[i]] = 1

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(SEQ_LENGTH, NUM_NOTES)))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(NUM_NOTES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

print("Обучаем модель...")
model.fit(X_oh, y_oh, epochs=EPOCHS, batch_size=BATCH_SIZE)

def sample_note(preds, temperature=1.0):
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    note = np.random.choice(range(NUM_NOTES), p=preds)
    return note

print("Генерируем мелодию...")
generated = list(X[0])

for _ in range(200):
    x_pred = np.zeros((1, SEQ_LENGTH, NUM_NOTES))
    for t in range(SEQ_LENGTH):
        note = generated[-SEQ_LENGTH + t]
        x_pred[0, t, note] = 1
    preds = model.predict(x_pred, verbose=0)[0]
    next_note = sample_note(preds, temperature=1.0)
    generated.append(next_note)

def save_melody_to_midi(sequence, filename="generated_dense_ainaudi_v2.mid", step_duration=0.4):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    time = 0
    for note_num in sequence:
        if note_num != 0:
            note = pretty_midi.Note(velocity=100, pitch=note_num, start=time, end=time+step_duration)
            piano.notes.append(note)
        time += step_duration
    pm.instruments.append(piano)
    pm.write(filename)

save_melody_to_midi(generated)
print("Готово! Файл saved:", "generated_dense_ainaudi_v2.mid")
