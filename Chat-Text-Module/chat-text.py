import json
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split

class ChatBot:
    def __init__(self):
        with open('/Users/mac/Documents/Chat-Bot-Telegram/dadrah_dataset1-100000_276342.json', 'r', encoding='utf-8') as f:
            data = json.load(f)


        print('sssscscscscscscs',data[3])
        for rowansewer in data :
            y_train = self.model.encode(rowansewer[2])

        rows = [[row[1], row[2][0] if isinstance(row[2], list) else row[2]] for row in data]
        df = pd.DataFrame(rows, columns=["question", "answer"])
 
       
        self.embeddings = np.load('/Users/mac/Documents/Chat-Bot-Telegram/embeddings.npy')
        self.embeddings = Normalizer().fit_transform(self.embeddings)

        print("✅ شکل embedding‌ها:", self.embeddings.shape)

        y = np.array(df['answer'])

        x_train, x_test, y_train, y_test = train_test_split(self.embeddings, y, test_size=0.2, random_state=42)

        self.model = Sequential([
            Dense(126, input_shape=(384,)),
            LeakyReLU(alpha=0.01),

            Dense(32),
            LeakyReLU(alpha=0.01),

            Dense(126),
            LeakyReLU(alpha=0.01),

            Dense(384, activation='sigmoid') 
        ])

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

if __name__ == "__main__":
    chatbot = ChatBot()
