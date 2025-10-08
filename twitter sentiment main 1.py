import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

data = pd.read_csv("Twitter_Data.csv")

data['clean_text'] = data['text'].astype(str).str.replace(r'[^a-z\s]', '', regex = True).str.lower()

sentiment_map = {'neutral': 0,
                 'negative': -1,
                 'positive':1
                 }

data['category'] = data['sentiment'].map(sentiment_map)

X = data['clean_text']
y = data['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

vocab_size = 5000
max_len = 100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen = max_len)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen = max_len)

labelencoder = LabelEncoder()
y_train_encoded = labelencoder.fit_transform(y_train)
y_test_encoded = labelencoder.fit_transform(y_test)

num_classes = len(labelencoder.classes_)
y_train_cat = tf.keras.utils.to_categorical(y_train_encoded, num_classes = num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test_encoded, num_classes = num_classes)

model = Sequential([
    Embedding(vocab_size, 100, input_length = max_len),
    LSTM(128, dropout = 0.2, recurrent_dropout = 0.2),
    Dense(64, activation = 'relu'),
    Dropout(0.5),
    Dense(num_classes, activation = 'softmax')])

model.compile(optimizer ='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train_pad, y_train_cat, validation_data =(X_test_pad, y_test_cat), epochs = 5, batch_size = 32)
model.save('sentiment_model.keras')

dump(tokenizer, 'tokenizer.joblib')
dump(labelencoder, 'labelencoder.joblib')

print("completed")














