import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Page configuration
st.set_page_config(page_title="Weather Classifier", layout="wide")

# Title and description
st.title("🌦️ KNN Weather Classification")
st.markdown("""
This app uses K-Nearest Neighbors (KNN) to classify weather conditions 
based on temperature and humidity levels.
""")

# Training data
x = np.array([
    [30, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [28, 75]
])
y = np.array([0, 1, 0, 0, 1, 1])

# Label mapping
label_map = {
    0: "Sunny",
    1: "Rainy"
}

# Custom KNN implementation (no sklearn needed!)
class SimpleKNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.x_train = None
        self.y_train = None
    
    def fit(self, x, y):
        self.x_train = x
        self. y_train = y
        return self
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, x):
        predictions = []
        for sample in x:
            distances = [self._euclidean_distance(sample, x_train) for x_train in self. x_train]
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)
        return np.array(predictions)
    
    def predict_proba(self, x):
        probabilities = []
        for sample in x:
            distances = [self._euclidean_distance(sample, x_train) for x_train in self.x_train]
            k_indices = np. argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            
            # Count occurrences
            sunny_count = np.sum(k_labels == 0)
            rainy_count = np.sum(k_labels == 1)
            
            # Calculate probabilities
            proba = np.array([sunny_count / self.n_neighbors, rainy_count / self.n_neighbors])
            probabilities.append(proba)
        return np.array(probabilities)

# Train the model
knn = SimpleKNN(n_neighbors=3)
knn.fit(x, y)

# Sidebar for user input
st.sidebar.header("📊 Input Parameters")
temperature = st.sidebar.slider("Temperature (°C)", min_value=20, max_value=35, value=26, step=1)
humidity = st.sidebar.slider("Humidity (%)", min_value=50, max_value=90, value=78, step=1)

# Make prediction
new_weather = np.array([[temperature, humidity]])
pred = knn.predict(new_weather)[0]
pred_proba = knn.predict_proba(new_weather)[0]

# Display prediction result
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Prediction Result")
weather_label = label_map[pred]
confidence = pred_proba[pred] * 100

# Color based on prediction
if pred == 0:
    st.sidebar.success(f"**Weather:  {weather_label}** ☀️")
else:
    st.sidebar.info(f"**Weather: {weather_label}** 🌧️")

st.sidebar.metric("Confidence", f"{confidence:.1f}%")

# Main content - Create visualization
col1, col2 = st. columns(2)

with col1:
    st.subheader("📈 Classification Visualization")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot training data
    ax.scatter(x[y==0, 0], x[y==0, 1], color="orange", label="Sunny", s=150, edgecolor="k", alpha=0.7)
    ax.scatter(x[y==1, 0], x[y==1, 1], color="blue", label="Rainy", s=150, edgecolor="k", alpha=0.7)
    
    # Plot new prediction
    colors = ["orange", "blue"]
    ax.scatter(new_weather[0, 0], new_weather[0, 1],
               color=colors[pred], marker="*", s=800, edgecolor="black", 
               label=f"New day:  {weather_label}", zorder=5)
    
    ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Humidity (%)", fontsize=12, fontweight="bold")
    ax.set_title("Weather Classification Model", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 35)
    ax.set_ylim(50, 90)
    
    st.pyplot(fig)

with col2:
    st.subheader("📋 Model Information")
    
    st.write("**Training Data Summary:**")
    st.write(f"- Total samples: {len(x)}")
    st.write(f"- Sunny days: {np.sum(y==0)}")
    st.write(f"- Rainy days: {np.sum(y==1)}")
    st.write(f"- K-neighbors: 3")
    
    st.markdown("---")
    st.write("**Current Input:**")
    st.write(f"- Temperature: **{temperature}°C**")
    st.write(f"- Humidity: **{humidity}%**")
    
    st.markdown("---")
    st.write("**Prediction Details:**")
    col_sunny, col_rainy = st. columns(2)
    with col_sunny:
        st. metric("Sunny Probability", f"{pred_proba[0]*100:.1f}%")
    with col_rainy: 
        st.metric("Rainy Probability", f"{pred_proba[1]*100:.1f}%")

# Footer
st.markdown("---")
st.caption("🔬 Built with Streamlit | KNN Weather Classification Model")
