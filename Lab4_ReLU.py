import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Neural Network Intelligence: ReLU App",
    page_icon="🧠",
    layout="wide"
)

# =============================
# Sidebar: Parameter Control
# =============================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("### 1. Visualization Settings")
x_min = st.sidebar.slider("Min x value", -10.0, 0.0, -5.0)
x_max = st.sidebar.slider("Max x value", 0.0, 10.0, 5.0)
num_points = st.sidebar.slider("Number of points", 10, 500, 100)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Model Complexity")
neurons = st.sidebar.slider("Hidden Layer Neurons", 1, 50, 15)
epochs = st.sidebar.slider("Epochs", 50, 500, 200)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1], value=0.01)

# =============================
# Header & Introduction
# =============================
st.title("Neural Network Architecture: ReLU Activation")
st.write(
    """
    This application explores the **Rectified Linear Unit (ReLU)**, a critical component in 
    modern artificial intelligence. We demonstrate how this mathematical rule enables a 
    neural network to solve complex, non-linear data problems by introducing piecewise 
    linear transformations into its decision-making process.
    """
)

# =============================
# Section 1: Interactive ReLU Visualization
# =============================
st.markdown("---")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📈 Interactive ReLU Curve")
    x_val = np.linspace(x_min, x_max, num_points)
    y_val = np.maximum(0, x_val)

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(x_val, y_val, color='#1f77b4', linewidth=2, label="ReLU(x)")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Input", fontsize=7)
    ax.set_ylabel("Output", fontsize=7)
    ax.tick_params(labelsize=6)
    st.pyplot(fig, use_container_width=False)

with col2:
    st.subheader("⚙️ Mathematical Framework")
    st.latex(r"f(x) = \max(0, x)")
    st.info("""
        **Core Technical Advantages:**
        * **Non-Linear Mapping:** Transforms linear inputs into adaptive representations to approximate complex functions.
        * **Computational Velocity:** Accelerates training by using simple thresholding instead of exponential math.
        * **Gradient Stability:** Mitigates vanishing gradients by maintaining a constant gradient for positive inputs.
        * **Feature Sparsity:** Improves efficiency by deactivating neurons with negative inputs.
        """)

# =============================
# Section 2: Data-Driven Neural Network Model
# =============================
st.markdown("---")
st.subheader("🤖 Live Model Training: Solving a Curve")
st.write("Below, we generate a dataset with a non-linear relationship. Watch the network use ReLU to adapt.")

X_np = np.linspace(-3, 3, 40).reshape(-1, 1)
y_np = X_np ** 2 + np.random.normal(0, 0.4, X_np.shape)
X_torch = torch.from_numpy(X_np).float()
y_torch = torch.from_numpy(y_np).float()

model = nn.Sequential(
    nn.Linear(1, neurons),
    nn.ReLU(),
    nn.Linear(neurons, 1)
)

if st.button('🚀 Execute Training Process'):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    progress_bar = st.progress(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(X_torch)
        loss = criterion(prediction, y_torch)
        loss.backward()
        optimizer.step()
        progress_bar.progress((epoch + 1) / epochs)

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        fig_res, ax_res = plt.subplots(figsize=(3, 2))
        ax_res.scatter(X_np, y_np, s=8, color='gray', alpha=0.6, label='Data')
        with torch.no_grad():
            y_pred = model(X_torch).numpy()
        ax_res.plot(X_np, y_pred, color='red', linewidth=1.5, label='NN Prediction')
        ax_res.set_title(f"Fit achieved using {neurons} Neurons", fontsize=8)
        ax_res.tick_params(labelsize=6)
        ax_res.legend(prop={'size': 5})
        st.pyplot(fig_res, use_container_width=False)

    with res_col2:
        st.success(f"Training Complete!")
        st.metric("Final Training Loss (MSE)", f"{loss.item():.4f}")  # Added professional metric

        st.info("""
        **Understanding the Training Process:**
        * **Epochs:** Full passes of the dataset to iteratively refine internal parameters.
        * **Optimization (Adam):** The engine that adjusts weights to reduce error based on gradients.
        * **Learning Rate:** The step size for the optimizer; determines how aggressively weights are updated.
        * **Loss (MSE):** The error between prediction and data; the goal is to drive this toward zero.
        * **Neural Complexity:** Number of neurons determines the capacity to learn non-linear patterns.
        """)

st.markdown("---")
st.caption("BSD3513 Introduction to Artificial Intelligence | Lab 4 – Neural Networks")