import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv('spam.csv', encoding='latin-1').iloc[:, [0, 1]]
df.columns = ['Category', 'Message']


df['spam'] = df['Category'].apply(lambda x: 1 if x.lower() == 'spam' else 0)


df['Message'] = df['Message'].str.lower()


df['Message'] = df['Message'].apply(lambda x: re.sub(r'\W', ' ', x))


stop_words = set(stopwords.words('english'))
df['Message'] = df['Message'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

lemmatizer = WordNetLemmatizer()
df['Message'] = df['Message'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))

df = df.drop_duplicates()


df.to_csv('cleaned_spam.csv', index=False)

@st.cache_resource
def load_model():
    
    df = pd.read_csv('spam.csv', encoding='latin-1').iloc[:, [0, 1]]
    df.columns = ['Category', 'Message']

   
    df['spam'] = df['Category'].apply(lambda x: 1 if x.lower() == 'spam' else 0)

    
    X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], test_size=0.25, random_state=42)

    
    vectorizer = CountVectorizer(binary=True)  
    X_train_count = vectorizer.fit_transform(X_train)
    X_test_count = vectorizer.transform(X_test) 

    
    model = MultinomialNB()
    model.fit(X_train_count, y_train)

    return model, vectorizer, X_test_count, X_test, y_test


model, vectorizer, X_test_count, X_test, y_test = load_model()


st.title("📩 Spam Detector App")
st.markdown("Enter a message to check if it's **Spam or Not Spam**.")


user_input = st.text_area("Enter your message here:", height=150)

if st.button("Check Message"):
    if user_input.strip():  
        input_count = vectorizer.transform([user_input])
        prediction = model.predict(input_count)[0]
        st.success("🚀 **Spam!**" if prediction == 1 else "✅ **Not Spam!**")
    else:
        st.warning("Please enter a message.")


st.write(f"📊 **Model Accuracy:** {model.score(X_test_count, y_test) * 100:.2f}%")
