# delivery-time-prediction
Delivery time prediction using PyTorch
# 🚚 Delivery Time Prediction using PyTorch

This project implements an **end-to-end Machine Learning pipeline** to predict **delivery time (in minutes)** using a neural network built with **PyTorch**.

The focus of this project is not just model building, but **correct data preprocessing, normalization, training, and inference consistency**, which are critical in real-world ML systems.

---

## 📌 Problem Overview

Given delivery-related inputs such as:
- Distance of delivery
- Time of the day
- Whether it is a weekend
- Whether it falls under rush hour

the model predicts the **estimated delivery time**.

---

## 🧠 Features Used

| Feature | Description |
|------|------------|
| Distance | Delivery distance in miles |
| Time of Day | Time in 24-hour format (e.g., 14 = 2 PM) |
| Weekend | 1 if weekend, 0 if weekday |
| Rush Hour | Engineered feature derived from time & weekday |

---

## 🏗️ Model Architecture

A **fully connected neural network** is used:

- **Input layer:** 4 features  
- **Hidden layers:**
  - Linear(4 → 64) + ReLU
  - Linear(64 → 32) + ReLU
- **Output layer:**
  - Linear(32 → 1)

**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Stochastic Gradient Descent (SGD)

---

## 🔄 Data Preprocessing Pipeline

1. Raw data extracted from a Pandas DataFrame
2. Feature engineering:
   - Rush hour detection based on time and weekday
3. Normalization:
   - Distance and time normalized using **training mean & standard deviation**
   - Binary features (weekend, rush hour) kept unchanged
4. All features concatenated into a single tensor

⚠️ **Key ML Principle Applied:**  
The **same normalization statistics from training** are reused during prediction to ensure consistency.

---

## 🧪 Training Process

- Model is trained for multiple epochs
- Loss values are monitored to verify convergence
- Training and inference pipelines are clearly separated

---

## 🔮 Inference (Prediction)

For new delivery inputs:
1. Raw inputs are converted to tensors
2. Features are normalized using **training statistics**
3. Prediction is performed using `torch.no_grad()` for efficiency

### Example Prediction
