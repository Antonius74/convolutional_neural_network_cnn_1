import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Scarico il dataset MNIST direttamente da Keras
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalizzazione dei pixel: da valori 0-255 a 0-1
X_train_full = X_train_full.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Ridimensionamento per aggiungere la dimensione del canale (monocromatico)
X_train_full = X_train_full.reshape(X_train_full.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Codifica one-hot delle etichette
y_train_full = to_categorical(y_train_full, 10)
y_test = to_categorical(y_test, 10)

# Divisione del training set in training e validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

# Data augmentation per aumentare il numero di esempi di training
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# Creazione del modello CNN
model = Sequential([
    # Primo blocco convoluzionale
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    # Secondo blocco convoluzionale
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Terzo blocco convoluzionale
    Conv2D(64, (3, 3), activation='relu'),
    
    # Strato di appiattimento per la connessione con i layer fully connected
    Flatten(),
    
    # Strati fully connected con dropout per ridurre overfitting
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compilazione del modello
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Addestramento del modello con data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    epochs=15,
    validation_data=(X_val, y_val),
    verbose=1
)

# Valutazione del modello sul test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

# Predizioni sul test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calcolo delle metriche di valutazione
print("=" * 50)
print("VALUTAZIONE DEL MODELLO CNN")
print("=" * 50)
print(f"Accuracy sul test set: {test_accuracy:.4f}")
print(f"Loss sul test set: {test_loss:.4f}")
print("\nMatrice di confusione:")
print(confusion_matrix(y_true_classes, y_pred_classes))
print("\nRapporto di classificazione:")
print(classification_report(y_true_classes, y_pred_classes))

# Grafico dell'accuracy durante l'addestramento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy durante l\'addestramento')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss durante l\'addestramento')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Salvataggio del modello
model.save('mnist_cnn_model.h5')
print("\nModello salvato come 'mnist_cnn_model.h5'")

# Esempio di predizione su un'immagine del test set
sample_index = 0
sample_image = X_test[sample_index]
sample_prediction = model.predict(sample_image.reshape(1, 28, 28, 1))
predicted_class = np.argmax(sample_prediction)
true_class = np.argmax(y_test[sample_index])

print(f"\nEsempio di predizione:")
print(f"Classe vera: {true_class}")
print(f"Classe predetta: {predicted_class}")
print(f"Probabilità: {np.max(sample_prediction):.4f}")