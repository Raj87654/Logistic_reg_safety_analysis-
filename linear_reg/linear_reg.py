import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

X = np.linspace(1, 20, 50)

true_m = 3
true_b = 5

Y = true_m * X + true_b + np.random.randn(50) * 3

m = np.random.randn()
b = np.random.randn()

learning_rate = 0.001
epochs = 200

loss_history = []


# -----------------------------
# 3. Training Loop
# -----------------------------

for epoch in range(epochs):

    # Prediction
    Y_pred = m * X + b

    # Loss (Mean Squared Error)
    loss = np.mean((Y - Y_pred) ** 2)
    loss_history.append(loss)

    # Gradients
    dm = -2 * np.mean(X * (Y - Y_pred))
    db = -2 * np.mean(Y - Y_pred)

    # Parameter Update
    m = m - learning_rate * dm
    b = b - learning_rate * db

    # Print progress
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.2f} | m: {m:.2f} | b: {b:.2f}")


print("\nFinal Learned Parameters")
print("m =", m)
print("b =", b)


# -----------------------------
# 4. Visualization
# -----------------------------

plt.figure(figsize=(10,6))

# Actual data
plt.scatter(X, Y, label="Actual Data")

# Predicted line
plt.plot(X, m*X + b, label="Learned Line", linewidth=3)

plt.xlabel("Distance (km)")
plt.ylabel("Battery Usage (%)")
plt.title("Drone Battery Prediction Using Linear Regression")
plt.legend()

plt.show()


# -----------------------------
# 5. Loss Curve Visualization
# -----------------------------

plt.figure(figsize=(10,6))

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Decreasing During Training")

plt.show()