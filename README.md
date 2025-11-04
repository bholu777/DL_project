# Handwritten Digit Classifier: An ANN vs. CNN Comparison
This project builds and compares two neural networks: a simple Artificial Neural Network(ANN) and a deep Convoluted Neural Network(CNN), on the MNIST handwritten digits dataset. The goal was to not just build a classifier, but to demonstrate why CNNs are better at analyzing image-based data. 
 
 **Dataset**: [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) 
 
## 1. The Baseline: Artificial Neural Network (ANN)
First, I established a simple, fully-connected ANN with Dense layers.

**Architecture**: `Flatten(input_shape=(28, 28)) -> Dense(128, activation='relu') -> Dense(10, activation='softmax')`

**Final Accuracy**: 97.8% (at 20 epochs)

### Training & Validation Plots:

These graphs show a very positive training process.

#### Accuracy vs. Epochs:

<img width="600" height="500" alt="Screenshot 2025-11-04 140923" src="https://github.com/user-attachments/assets/e15ab44d-1147-4ab9-b10b-60ddd45d3aec" />

#### Loss vs. Epochs:

<img width="600" height="500" alt="Screenshot 2025-11-04 140856" src="https://github.com/user-attachments/assets/1bdbe203-0681-42c6-b321-dcc5043537fd" />

#### Confusion matrix:

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/56a26b61-4990-4649-97ca-f83ea3978501" />

### Analysis: 
While 97.8% is a good score, the model's confusion matrix revealed a clear flaw: it struggled with spatially similar digits. For example, it frequently misclassified 9's as 4's (12 errors) and 9's as 7's (8 errors). This is because the Flatten layer destroys all 2D spatial information before the model can even see it.

## 2. The Solution: Convolutional Neural Network (CNN)
To solve this spatial problem, I built a deep CNN designed to learn 2D patterns. To combat overfitting in this deeper model, I included BatchNormalization and Dropout layers in each block.

### Final Architecture:

Layer 1: `Conv2D(64) -> BatchNormalization -> relu -> MaxPooling2D -> Dropout(0.25)`

Layer 2: `Conv2D(64) -> BatchNormalization -> relu -> MaxPooling2D -> Dropout(0.25)`

Layer 3: `Conv2D(64) -> BatchNormalization -> relu -> MaxPooling2D -> Dropout(0.25)`

Classifier: `Flatten -> Dense(64) -> BatchNormalization -> relu -> Dropout(0.5)`

Classifier: `Dense(32) -> BatchNormalization -> relu -> Dropout(0.5)`

Output: `Dense(10, 'softmax')`

## 3. CNN Results and Analysis
This model was a significant improvement.

**Final Accuracy**: 98.65% (at 20 epochs)

### Training & Validation Plots:

These graphs show a very healthy training process.

#### Accuracy vs. Epochs:

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/22ae4f05-70ff-456d-a0ea-1fdcf8ec6cbe" />

#### Loss vs. Epochs:

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/7882495c-d7eb-4e12-95ba-f838d57187d8" />
 
**Key Observation**: The validation accuracy (orange line) is consistently higher than the training accuracy (blue line). This is a classic and positive sign of Dropout working correctly. The model is "nerfed" during training (as neurons are randomly dropped) but uses its full power during validation, resulting in better performance. This shows the model is generalizing well and is not overfitting.

#### Confusion Matrix:

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/82e42609-160d-4b46-a805-a6e0cff67514" />

### Analysis: 
The CNN's performance is a clear success. The errors that plagued the ANN have been almost completely solved:

**'9' vs. '4' Errors**: Dropped from 12 to 4.

**'9' vs. '7' Errors**: Dropped from 8 to 2.

The model's primary remaining confusion (e.g., '2' vs. '7', with 11 and 13 errors) is a much harder, well-known problem, but the original spatial issues were fixed.

## Conclusion

The CNN (98.65%) clearly outperformed the ANN (97.8%). By using Conv2D layers to analyze the 2D image structure before flattening, the model was able to learn robust spatial features (like loops vs. open lines), which directly solved the ANN's main failure points.
