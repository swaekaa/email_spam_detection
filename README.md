# 📧 Email Spam Detection Web App 🚫

## Welcome to the Email Spam Detection Web App! This simple and interactive app allows you to check whether the email you've received is spam or not. It uses a machine learning model based on the Naive Bayes algorithm to classify emails as either "Spam" or "Not Spam" (Ham).


## 🌟 Key Features
- Spam Detection: Paste the content of your email to check if it's spam. 📩
- Easy-to-Use Interface: Just type or paste your email text and get results instantly! ⏱️
- Fast & Efficient: Real-time spam detection without any complex setup. ⚡



## ⚙️ Technologies Used
- Streamlit: For building the interactive web interface. 🌐
- Naive Bayes Classifier: A machine learning algorithm to classify emails. 🤖
- Python: The programming language used to build the backend of the app. 🐍



## 🚀 How to Get Started

### 1. Clone the Repository
 Start by cloning the project to your local machine:
 git clone 
 ```bash
https://github.com/swaekaa/email_spam_detection.git
```

### 2. Install Required Packages
Make sure you have Python 3.7+ installed. Then, install the necessary packages using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the Web App
Once the dependencies are installed, you can start the app by running:
```bash
streamlit run app.py
```
This will launch the app in your default web browser. If it doesn't open automatically, go to http://localhost:8501 in your browser. 🌍

### 4. How to Use
- Paste the content of the email you received in the provided text box. ✍️
- Click on the "Check if Spam" button. 🖱️
- Instantly, you'll see whether the email is classified as Spam or Not Spam (Ham). 📊



## 🔄 Retraining the Model
You can also retrain the model using your own dataset. Here's how you can do it:



### Steps to Retrain the Model
- Place your email dataset in the data/ folder. 📂
- Run the training script:
```bash
python train_model.py
```
This will train the model with your dataset, and the updated model will be saved for use. 🔄







## 🤝 Contributing
We welcome contributions! If you find any bugs or want to add features, feel free to fork the repo and submit a pull request. 🔧



## Thank you for using the Email Spam Detection Web App! 📨🚫
