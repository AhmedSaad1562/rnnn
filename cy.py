import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


text = "I love cyber security"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

X = [sequences[:3]] 
y = [sequences[3]]   

X = np.array(X)
y = np.array(y)

vocab_size = len(tokenizer.word_index) + 1
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=3))
model.add(SimpleRNN(10))
model.add(Dense(vocab_size, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=0)


prediction = model.predict(X)
predicted_word = tokenizer.index_word[np.argmax(prediction)]
print(f"Predicted word: {predicted_word}")