import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load data
mat = scipy.io.loadmat('Xtrain.mat')
data = mat['Xtrain'].flatten().reshape(-1, 1)

# Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Create supervised datase
def create_dataset(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps].flatten())
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

# Forward pass over different n_steps
best_mse = float('inf')
best_n = None
results = {}

for n_steps in range(2, 21):
    X, y = create_dataset(scaled, n_steps)
    
    # Train-test split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Build simple FNN model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(n_steps,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    # Evaluate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    results[n_steps] = mse

    print(f"n_steps = {n_steps}, MSE = {mse:.6f}")

    if mse < best_mse:
        best_mse = mse
        best_n = n_steps

print(f"\n✅ Best number of lookback steps: {best_n} (MSE = {best_mse:.6f})")
