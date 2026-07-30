"""
solve classification problem (sentiment analysis in NLP) with RNN
duygu analizi -> bir cümlenin etiketlenmesi(negatif ve pozitif)
restaurant yorumları değerlendirmesi
"""

# import libraries
import numpy as np
import pandas as pd

from gensim.models import Word2Vec   # metin temsili

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer  
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# create dataset
data = {
    "text": [
        "Yemekler çok güzeldi.",
        "Garsonlar çok ilgiliydi.",
        "Restoranın ortamı harikaydı.",
        "Servis çok hızlıydı.",
        "Tatlılar enfesti.",
        "Yemekler çok pişmişti.",
        "Siparişimiz çok geç geldi.",
        "Garsonlar ilgisizdi.",
        "Masa temiz değildi.",
        "Fiyatlar çok yüksekti.",
        "Yemeklerin lezzeti gerçekten harikaydı.",
        "Personel çok güler yüzlü ve yardımseverdi.",
        "Siparişimiz çok kısa sürede geldi.",
        "Fiyatına göre oldukça kaliteli bir restorandı.",
        "Tatlılar tazeydi ve çok beğendik.",
        "Restoran oldukça temiz ve düzenliydi.",
        "Porsiyonlar doyurucuydu.",
        "Sunumlar çok şık hazırlanmıştı.",
        "Ailecek çok memnun kaldık.",
        "Kesinlikle tekrar geleceğimiz bir mekan.",
        "Yemekler soğuk servis edildi.",
        "Garsonlar siparişimizi yanlış getirdi.",
        "Bekleme süresi çok uzundu.",
        "Masalar yeterince temiz değildi.",
        "Yemeklerin tadı beklediğimiz gibi değildi.",
        "Fiyatlar verilen hizmete göre çok yüksekti.",
        "Çorbanın içinden yabancı bir cisim çıktı.",
        "Restoran çok kalabalık ve gürültülüydü.",
        "Tatlılar bayattı.",
        "Bir daha tercih etmeyeceğim bir restoran."        
    ],
    "label": [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative"        
    ]
}

df = pd.DataFrame(data)

#%% metin temizleme ve preprocessing: tokeniztain, padding, label encoding, train test split

#tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index

# padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen = maxlen)
print(X.shape)

#label encoding

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%% metin temsili:word embedding: word2vec

# 1. Metinleri kelimelerine ayır
sentences = [text.split() for text in data["text"]]

# 2. Word2Vec modelini eğit
word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1)

# 3. Embedding matrisini oluştur
embedding_dim = 50
# word_index, Tokenizer'dan gelen sözlük yapısıdır (+1 padding için eklenir)
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))

# 4. Eğitilen vektörleri matrise aktar
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]


# %% modelling: build, train ve test rnn modeli

model = Sequential()

# embedding
model.add(Embedding(input_dim = len(word_index) + 1, output_dim = embedding_dim, weights = [embedding_matrix], input_length=maxlen, trainable = False))

# RNN layer
model.add(SimpleRNN(50, return_sequences = False))

# output layer
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])

# train model
model.fit(X_train, y_train, epochs=10, batch_size = 2, validation_data=(X_test, y_test))

# evaluate rnn model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# %% cumle siniflandirma calismasi
def classify_sentence(sentence):
    
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen = maxlen)
    
    prediction = model.predict(padded_seq)
    
    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"
    
    return label

sentence = "Restaurant çok temizdi ve yemekler çok güzeldi"

result = classify_sentence(sentence)
print(f"Result: {result}")

























