# Reti Neurali Convoluzionali (CNN)

## 1. Introduzione e Teoria

Le Reti Neurali Convoluzionali sono un tipo speciale di rete neurale particolarmente efficace per l'elaborazione di dati con struttura a griglia, come le immagini. Pensale come un sistema di visione artificiale che impara a riconoscere pattern visivi in modo gerarchico.

### Come funziona una CNN

Immagina di voler insegnare a un computer a riconoscere se in una foto c'è un gatto. Una CNN affronta questo problema attraverso diversi strati specializzati:

**Strato Convoluzionale**: È come avere una piccola lente d'ingrandimento che scorre sull'immagine per cercare caratteristiche specifiche. Questa "lente" è chiamata **filtro** o **kernel**.

$$
Feature\ Map = Input \ast Kernel + Bias
$$

Dove $\ast$ rappresenta l'operazione di convoluzione.

**Esempio pratico**: Se abbiamo un'immagine 3x3 e un filtro 2x2:

$$
Input = \begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 1
\end{bmatrix},\quad
Kernel = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

Il risultato della convoluzione sarebbe:

$$
Output = (1\times1 + 0\times0 + 0\times0 + 1\times1) = 2
$$

**Strato di Pooling**: Riduce le dimensioni dell'immagine mantenendo le caratteristiche più importanti. È come comprimere una foto mantenendo solo i dettagli salienti.

**Strati Fully Connected**: Alla fine della rete, questi strati prendono le caratteristiche estratte e le usano per fare la classificazione finale.

## 2. Dati Utilizzati (Input/Output)

### Dataset MNIST

Il dataset MNIST è una raccolta di 70.000 immagini di cifre scritte a mano (0-9). È il "Hello World" del riconoscimento di immagini.

**Struttura dell'input**:
- **Dimensione**: 28x28 pixel
- **Canali**: 1 (immagini in scala di grigi)
- **Valori dei pixel**: da 0 (nero) a 255 (bianco)
- **Forma del dato**: (28, 28, 1)

**Esempio numerico di un'input** (immagine ridotta 5x5 per semplicità):

$$
\begin{bmatrix}
0 & 0 & 255 & 0 & 0 \\
0 & 255 & 0 & 255 & 0 \\
255 & 0 & 0 & 0 & 255 \\
0 & 255 & 0 & 255 & 0 \\
0 & 0 & 255 & 0 & 0
\end{bmatrix}
$$

Questa potrebbe rappresentare uno zero o un otto molto stilizzato.

**Output atteso**:
- **Tipo**: Classificazione multiclasse
- **Range**: Numeri da 0 a 9
- **Forma**: Vettore di 10 elementi (one-hot encoding)

Esempio per il numero "3":
$$
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
$$

## 3. Analisi del Codice

### Preprocessing dei Dati

```python
# Normalizzazione dei pixel
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
```

I valori dei pixel originali (0-255) vengono scalati tra 0 e 1. Questo migliora la stabilità numerica durante l'addestramento.

$$
Pixel_{normalizzato} = \frac{Pixel_{originale}}{255}
$$

```python
# One-hot encoding delle labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

Convertiamo le etichette da numeri interi a vettori binari. Per esempio, il numero 3 diventa:

$$
3 \rightarrow [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
$$

### Architettura della CNN

**Strato Convoluzionale 1**:
```python
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
```
- 32 filtri di dimensione 3x3
- Funzione di attivazione ReLU: $f(x) = max(0, x)$
- Input: immagini 28x28x1

**Pooling Layer**:
```python
MaxPooling2D((2, 2))
```
Riduce la dimensione spaziale prendendo il valore massimo in finestre 2x2.

**Strato Fully Connected**:
```python
Dense(128, activation='relu')
Dense(10, activation='softmax')
```

La funzione softmax converte i logit in probabilità:

$$
Softmax(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
$$

### Addestramento del Modello

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

**Funzione di Loss** (Cross-Entropy):
$L = -\sum_{i=1}^{10} y_i \cdot \log(\hat{y}_i)$

Dove $y_i$ è la label vera e $\hat{y}_i$ è la predizione.

**Ottimizzatore Adam**: Adaptive Moment Estimation, un algoritmo di ottimizzazione avanzato che adatta il learning rate per ogni parametro.

### Valutazione delle Prestazioni

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
```

La accuracy è calcolata come:

$$
Accuracy = \frac{Numero\ di\ predizioni\ corrette}{Numero\ totale\ di\ campioni}
$$

### Predizione

```python
predictions = model.predict(x_test)
```

Le predizioni sono vettori di probabilità. Per ottenere la classe predetta:

$$
Classe\ Predetta = \arg\max_{i} prediction_i
$$

Questo significa scegliere la classe con la probabilità più alta nel vettore di output.
