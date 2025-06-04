import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
import nest_asyncio
import logging
import os
from dotenv import load_dotenv


nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

with open('/content/dadrah_dataset1-100000_276342.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pairs = [(row[1], row[2][0] if isinstance(row[2], list) else row[2]) for row in data]
texts = [f"سوال: {q} پاسخ: {a}" for q, a in pairs]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(3, len(token_list)):
        input_sequences.append(token_list[:i])

max_seq_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

x = input_sequences[:, :-1]
y = input_sequences[:, -1]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_seq_len - 1),
    tf.keras.layers.LSTM(150),
    tf.keras.layers.Dense(total_words, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=10, verbose=1)

def generate_answer(seed_text, max_words=20):
    for _ in range(max_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted)
        output_word = tokenizer.index_word.get(predicted_index, '')
        if output_word == '':
            break
        seed_text += ' ' + output_word
        if output_word == 'پاسخ':
            break
    return seed_text


async def handle_message(update, context):
    user_input = update.message.text
    response = generate_answer(f"سوال: {user_input}")
    await update.message.reply_text(response)

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("🤖 ربات آماده است...")
app.run_polling()
