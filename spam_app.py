import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('spam.csv', encoding='latin-1')  
df = df.iloc[:, [0, 1]]  
df.columns = ['Category', 'Message']  
df['spam'] = df['Category'].apply(lambda x: 1 if x.lower() == 'spam' else 0)  


X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], test_size=0.25, random_state=42)


vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_count, y_train)


st.title("📩 SPAM DETECTION APP")
st.write("Enter a message below to check if it's spam or not.")

user_input = st.text_area("Enter Message:",height=250)
if st.button("Predict"):
    if user_input.strip():
        input_count = vectorizer.transform([user_input])
        prediction = model.predict(input_count)[0]
        st.success("🚀 Prediction: **Spam**" if prediction == 1 else "✅ Prediction: **Not Spam**")
    else:
        st.warning("Please enter a message.")

# Display model accuracy
st.write(f"📊 **Model Accuracy:** {model.score(X_test_count, y_test) * 100:.2f}%")

