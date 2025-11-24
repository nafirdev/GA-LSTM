
# GA-LSTM: Time Series Forecasting with Genetic Algorithm Optimization

GA-LSTM is an intelligent time-series forecasting framework that combines **Long Short-Term Memory (LSTM)** neural networks with a **Genetic Algorithm (GA)** to automatically optimize hyperparameters.  
Instead of manually tuning your model, the GA searches for the best configuration to achieve the lowest prediction error.

---

## ✨ Features
- **Hybrid Intelligence:** LSTM for sequential learning + GA for hyperparameter optimization.  
- **Automated Hyperparameter Search:**  
  Optimizes number of layers, units, learning rate, window size, and more.  
- **Domain-Independent:** Works with financial data, IoT data, energy demand, weather series, and any other sequential dataset.  
- **Lightweight Implementation:** Built using Keras and TensorFlow.

---

## 🚀 Quick Start

### Requirements
Install the required libraries:
```bash
pip install numpy pandas tensorflow scikit-learn
````

### Run the Model

```bash
python GA-LSTM.py
```

This will execute the genetic algorithm, search for the best LSTM configuration, train the optimized model, and display the results.

---

## 🧠 How It Works

1. **Define Search Space**
   GA samples and evolves hyperparameters such as:

   * Number of LSTM units
   * Number of LSTM layers
   * Learning rate
   * Window size (input time steps)

2. **Generate Initial Population**
   Random hyperparameter sets are created.

3. **Fitness Evaluation**
   Each candidate configuration trains an LSTM model and is evaluated using MSE (or other metrics).

4. **GA Operations**

   * **Selection** of the best candidates
   * **Crossover** to mix their hyperparameters
   * **Mutation** to introduce randomness

5. **Iterations Continue** until the best configuration emerges.

### Example Output

```
Best Configuration:
- Units: 120
- Layers: 2
- Learning Rate: 0.005
- Window Size: 30
Validation MSE: 0.00231
```

---

## 📁 Project Structure

```
GA-LSTM/
│
├── GA-LSTM.py        # Main implementation (GA + LSTM)
├── README.md         # Project documentation
└── (optional future files: saved models, plots, configs)
```

---

## 🎯 Intended Audience

This project is useful for:

* Data Scientists exploring automated neural network tuning
* Machine Learning Engineers working with time-series forecasting
* Researchers studying evolutionary algorithms + deep learning
* Anyone tired of manual hyperparameter tuning for LSTM models

---

## 🛠 Recommendations

* Prepare your dataset in sliding-window format (input → target).
* GPU acceleration is recommended for large populations or complex models.
* Visualization of GA progress (fitness per generation) can improve interpretability.

---

## 📜 License

This project is released under the **MIT License**, allowing free use, modification, and distribution.




