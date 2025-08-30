# 🏦 Customer Churn Prediction using Artificial Neural Networks (ANN)

A machine learning project that predicts customer churn in the banking sector using Artificial Neural Networks (ANN) built with TensorFlow/Keras.

## 📊 Project Overview

This project analyzes customer data to predict whether a bank customer will churn (leave the bank) or stay. Using deep learning techniques, we achieve accurate predictions that can help banks proactively retain customers.

### 🎯 Key Features

- **Deep Learning Model**: Sequential Neural Network with 3 layers
- **Interactive Web App**: Streamlit-based user interface for real-time predictions
- **Data Preprocessing**: Complete pipeline including encoding, scaling, and feature engineering
- **Model Deployment**: Ready-to-use prediction system with trained model

## 🚀 Live Demo

Try the live prediction app: [Run the Streamlit app locally]

## 📁 Project Structure

```
├── experiments.ipynb          # Main training notebook with step-by-step analysis
├── prediction.ipynb          # Prediction testing notebook
├── app.py                    # Streamlit web application (original)
├── app_fixed.py             # Enhanced Streamlit application
├── Churn_Modelling.csv      # Dataset
├── requirements.txt         # Python dependencies
├── model_architecture.json # Model architecture configuration
└── README.md               # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/nish0753/Churn-Prediction-using-ANN.git
cd Churn-Prediction-using-ANN
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app_fixed.py
```

## 📈 Model Architecture

Our ANN model consists of:

- **Input Layer**: 12 features (after preprocessing)
- **Hidden Layer 1**: 64 neurons with ReLU activation
- **Hidden Layer 2**: 32 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation (binary classification)

### Model Performance

- **Training Accuracy**: ~86%
- **Validation Accuracy**: ~84%
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

## 🔍 Data Features

The dataset includes the following customer attributes:

| Feature         | Description                                  | Type        |
| --------------- | -------------------------------------------- | ----------- |
| CreditScore     | Customer's credit score                      | Numerical   |
| Geography       | Customer's location (France, Germany, Spain) | Categorical |
| Gender          | Customer's gender                            | Categorical |
| Age             | Customer's age                               | Numerical   |
| Tenure          | Years as bank customer                       | Numerical   |
| Balance         | Account balance                              | Numerical   |
| NumOfProducts   | Number of bank products used                 | Numerical   |
| HasCrCard       | Has credit card (0/1)                        | Binary      |
| IsActiveMember  | Active membership status (0/1)               | Binary      |
| EstimatedSalary | Estimated salary                             | Numerical   |
| **Exited**      | **Target: Customer churned (0/1)**           | **Binary**  |

## 🧮 Data Preprocessing Pipeline

1. **Data Cleaning**: Remove irrelevant columns (RowNumber, CustomerId, Surname)
2. **Encoding**:
   - Label Encoding for Gender (Male=1, Female=0)
   - One-Hot Encoding for Geography (France, Germany, Spain)
3. **Feature Scaling**: StandardScaler for numerical features
4. **Train-Test Split**: 80% training, 20% testing

## 📊 Usage Examples

### Training the Model

```python
# Run the training notebook
jupyter notebook experiments.ipynb
```

### Making Predictions

```python
# Example prediction
customer_data = {
    'CreditScore': 650,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 35,
    'Tenure': 5,
    'Balance': 50000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 60000
}

# Use the Streamlit app or prediction notebook
```

## 🌐 Web Application Features

### Enhanced Streamlit App (`app_fixed.py`)

- **Interactive Input**: Sliders, selectboxes for all features
- **Real-time Predictions**: Instant churn probability calculation
- **Visual Results**: Color-coded risk assessment
- **Customer Insights**: Profile summary and recommendations
- **Error Handling**: Robust model loading and prediction pipeline

### Screenshots

[Add screenshots of your Streamlit app here]

## 📋 Requirements

```
tensorflow==2.16.1
streamlit==1.49.0
pandas==2.3.2
numpy==1.26.4
scikit-learn==1.7.1
matplotlib==3.10.5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Nishant**

- GitHub: [@nish0753](https://github.com/nish0753)
- LinkedIn: [Your LinkedIn Profile]

## 🙏 Acknowledgments

- Dataset source: [Kaggle - Bank Customer Churn Prediction](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)
- TensorFlow/Keras for deep learning framework
- Streamlit for creating the interactive web application
- Scikit-learn for preprocessing utilities

## 📚 What I Learned

- **Deep Learning**: Building and training neural networks for binary classification
- **Data Preprocessing**: Handling categorical variables, feature scaling, and encoding
- **Model Deployment**: Creating interactive web applications with Streamlit
- **Version Control**: Managing machine learning projects with Git
- **Error Handling**: Debugging model loading and compatibility issues

---

⭐ **Star this repository if you found it helpful!**
