import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load your dataset
data_dir = r"C:\Users\prade\OneDrive\Desktop\Major Project\Datasets\Deap\New_mat"

X = []
y_valence = []
y_arousal = []

# Initialize StandardScaler
scaler = StandardScaler()

for participant_id in range(1, 33):
    print(f"Loading data for participant {participant_id}")
    data = sio.loadmat(f"{data_dir}\\s{participant_id:02d}.mat")

    # Extract PPG, EEG, and GSR signals
    ppg_signal = data['data'][:, 38, :]
    eeg_signals = data['data'][:, 31, :]  # EEG signals from channels 1 to 32
    gsr_signal = data['data'][:, 35, :]  # GSR signal from channel 36

    # Extract statistical features from signals
    features_ppg = np.hstack([
        np.mean(ppg_signal, axis=1),
        np.std(ppg_signal, axis=1),
        np.min(ppg_signal, axis=1),
        np.max(ppg_signal, axis=1),
        np.ptp(ppg_signal, axis=1),  # Peak-to-Peak
        scipy.stats.skew(ppg_signal, axis=1),
        scipy.stats.kurtosis(ppg_signal, axis=1)
    ])

    features_eeg = np.hstack([
        np.mean(eeg_signals, axis=1),
        np.std(eeg_signals, axis=1),
        np.min(eeg_signals, axis=1),
        np.max(eeg_signals, axis=1),
        np.ptp(eeg_signals, axis=1),  # Peak-to-Peak
        scipy.stats.skew(eeg_signals, axis=1),
        scipy.stats.kurtosis(eeg_signals, axis=1)
    ])

    features_gsr = np.hstack([
        np.mean(gsr_signal, axis=1),
        np.std(gsr_signal, axis=1),
        np.min(gsr_signal, axis=1),
        np.max(gsr_signal, axis=1),
        np.ptp(gsr_signal, axis=1),  # Peak-to-Peak
        scipy.stats.skew(gsr_signal, axis=1),
        scipy.stats.kurtosis(gsr_signal, axis=1)
    ])

    # Stack features together
    participant_features = np.column_stack([features_ppg, features_eeg, features_gsr])

    # Normalize participant features
    normalized_features = scaler.fit_transform(participant_features)

    # Append to X
    X.append(normalized_features)

    # Append labels to y_valence and y_arousal
    y_valence.append(data['labels'][:, 0])
    y_arousal.append(data['labels'][:, 1])

# Convert lists to numpy arrays
X = np.array(X)
y_valence = np.array(y_valence)
y_arousal = np.array(y_arousal)

# Reshape data for LSTM input (samples, timesteps, features)
n_samples, n_timesteps, n_features = X.shape[0], 40, X.shape[2]
X_lstm = X.reshape(n_samples, n_timesteps, n_features)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_valence, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")