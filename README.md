# Handwritten Digit Recognition on Scanned Documents

A deep learning model that classifies handwritten digits from scanned document snippets using a Convolutional Neural Network (CNN). This project uses the **NIST SD19** dataset — more complex and real-world than MNIST — and is trained & evaluated using **TensorFlow** on **Google Colab**.

---

## Dataset

We use the [NIST Special Database 19 (SD19)](https://www.nist.gov/srd/nist-special-database-19) — a large dataset of handwritten digits and letters collected from forms.

- Subset used: `Digits (0–9)`
- Images resized to: `28x28` grayscale
- Total samples: ~53k
- Preprocessed using `ImageDataGenerator` with validation split (20%)

**Data Directory Structure:**
```bash
NIST SD1_digits/
├── 0/
├── 1/
├── 2/
├── ...
└── 9/
```

---

## Model Architecture

A simple yet effective CNN built with Keras:

```python
Conv2D(32) → MaxPooling2D → 
Conv2D(64) → MaxPooling2D →
Flatten → Dense(128) → Dropout(0.3) → Dense(10, softmax)
```
Loss: sparse_categorical_crossentropy

Optimizer: Adam

Input shape: (28, 28, 1)

## Training
```bash
model.fit(train_gen, epochs=10, validation_data=val_gen)
```
    Batch size: 32

    Epochs: 10

    Training handled via ImageDataGenerator with on-the-fly rescaling and validation spl
## Results
Metric	Score
Accuracy	98%
Macro Avg F1	0.98
Weighted Avg	0.98

    Confusion Matrix:

    Classification Report:
  ``` bash
                  precision    recall  f1-score   support

           0       0.97      0.99      0.98      1106
           1       0.99      0.99      0.99      1201
           2       0.99      0.97      0.98      1064
           3       0.98      0.97      0.98      1118
           4       0.98      0.98      0.98      1022
           5       0.99      0.97      0.98       920
           6       0.98      0.99      0.98      1052
           7       0.99      0.99      0.99      1117
           8       0.96      0.99      0.97      1038
           9       0.99      0.98      0.98      1024
```
## Preprocessing
``` bash
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```
Loaded with:
```bash
train_gen = datagen.flow_from_directory(
    path,
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    path,
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='sparse',
    batch_size=32,
    subset='validation',
    shuffle=False
)

```
## What’s Interesting

    This isn't MNIST — these digits come from real scanned forms, making them messier, varied, and harder to classify.

    Shows how well CNNs generalize with proper preprocessing and architecture.

    No advanced techniques (yet): this is a solid baseline for future experimentation.

## Author

Shreya Kumari
Computer Science Student, Cybersecurity & AI Enthusiast

