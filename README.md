# Customer Churn Prediction Using Artificial Neural Networks

A machine learning project that predicts customer churn using Artificial Neural Networks (ANN) with a professional Streamlit web application interface.

## Project Overview

This project implements a binary classification model to predict whether a bank customer will leave (churn) or stay with the bank based on various customer attributes. The model uses deep learning techniques with TensorFlow/Keras to achieve accurate predictions.

## Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Neural Network Model**: Deep learning implementation using TensorFlow/Keras
- **Web Application**: Interactive Streamlit interface for real-time predictions
- **Model Persistence**: Trained models and encoders saved for deployment
- **Professional Interface**: Clean, business-ready user interface

## Dataset

The project uses a bank customer dataset containing the following features:
- Customer demographics (age, gender, geography)
- Account information (credit score, balance, tenure)
- Product usage (number of products, credit card status)
- Activity metrics (active member status, estimated salary)

## Model Architecture

- **Input Layer**: 11 features after preprocessing
- **Hidden Layers**: Dense layers with ReLU activation
- **Output Layer**: Single neuron with sigmoid activation for binary classification
- **Optimization**: Adam optimizer with binary crossentropy loss

## Technical Stack

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Data preprocessing and model evaluation
- **Matplotlib**: Data visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nish0753/Churn-Prediction-using-ANN.git
cd Churn-Prediction-using-ANN
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the Jupyter notebook to train the model:
```bash
jupyter notebook experiments.ipynb
```

### Running the Web Application

Launch the Streamlit application:
```bash
streamlit run app.py
```

### Making Predictions

1. Open the web application in your browser
2. Input customer information in the sidebar
3. Click "Predict Churn" to get the prediction
4. View the probability and recommendation

## Project Structure

```
├── app.py                          # Streamlit web application
├── experiments.ipynb               # Model training notebook
├── prediction.ipynb                # Individual prediction testing
├── Churn_Modelling.csv            # Dataset
├── requirements.txt                # Python dependencies
├── model.h5                        # Trained neural network model
├── scaler.pkl                      # Feature scaler
├── label_encoder_gender.pkl        # Gender label encoder
├── onehot_encoder_geo.pkl          # Geography one-hot encoder
└── README.md                       # Project documentation
```

## Model Performance

The trained model achieves:
- High accuracy on test data
- Balanced precision and recall
- Robust performance across different customer segments

## Deployment

The application is optimized for deployment on:
- Streamlit Cloud
- Heroku
- AWS EC2
- Local environments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is available for educational and research purposes.

## Contact

For questions or suggestions, please open an issue in the repository.

## Acknowledgments

- Dataset providers for making the customer data available
- TensorFlow and Streamlit communities for excellent documentation
- Open source contributors who made this project possible