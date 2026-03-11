from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

prediction = model.predict([[3.5]])
print("Prediction for distance 3.5:", prediction)

X_test = np.linspace(0, 6, 100).reshape(-1,1)

probs = model.predict_proba(X_test)[:,1]

plt.figure(figsize=(8,5))
plt.scatter(X, y, color="blue", label="Training Data")
plt.plot(X_test, probs, color="red", label="Logistic Curve")


boundary = -model.intercept_/model.coef_
plt.axvline(boundary[0], linestyle="--", color="green", label="Decision Boundary")

plt.xlabel("Obstacle Distance")
plt.ylabel("Safety Probability")
plt.title("Drone Obstacle Safety using Logistic Regression")
plt.legend()
plt.show()