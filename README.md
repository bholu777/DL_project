# Handwritten Digit Classifier: An ANN vs. CNN Comparison
This project builds and compares two neural networks: a simple Artificial Neural Network(ANN) and a deep Convoluted Neural Network(CNN), on the MNIST handwritten digits dataset. The goal was to not just build a classifier, but to demonstrate why CNNs are better at analyzing image-based data. 
 
 <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> 
 
## 1. The Baseline: Artificial Neural Network (ANN)
First, I established a baseline using a simple, fully-connected ANN with Dense layers.

**Architecture**: `Flatten(input_shape=(28, 28)) -> Dense(128, activation='relu') -> Dense(10, activation='softmax')`

**Final Accuracy**: 97.89% (at 20 epochs)

**Analysis**: While 97.89% is a good score, the model's confusion matrix revealed a clear flaw: it struggled with spatially similar digits. For example, it frequently misclassified '9's as '4's (12 errors) and '9's as '7's (8 errors). This is because the Flatten layer destroys all 2D spatial information before the model can even see it.

## 2. The Solution: Convolutional Neural Network (CNN)
To solve this spatial problem, I built a deep CNN designed to learn 2D patterns. To combat overfitting in this deeper model, I included BatchNormalization and Dropout layers in each block.

**Final Architecture**:

Layer 1: `Conv2D(64) -> BatchNormalization -> relu -> MaxPooling2D -> Dropout(0.25)`

Layer 2: `Conv2D(64) -> BatchNormalization -> relu -> MaxPooling2D -> Dropout(0.25)`

Layer 3: `Conv2D(64) -> BatchNormalization -> relu -> MaxPooling2D -> Dropout(0.25)`

Classifier: `Flatten -> Dense(64) -> BatchNormalization -> relu -> Dropout(0.5)`

Classifier: `Dense(32) -> BatchNormalization -> relu -> Dropout(0.5)`

Output: `Dense(10, 'softmax')`

## 3. CNN Results and Analysis
This model was a significant improvement.

**Final Accuracy**: 98.71% (at 20 epochs)


### **Training & Validation Plots**

These graphs show a very healthy training process.

#### Accuracy vs. Epochs:

<img width="600" height="586" alt="Screenshot 2025-11-03 232336" src="https://github.com/user-attachments/assets/dce080f2-8662-459b-81a6-acaf42caf809" />

#### Loss vs. Epochs:

<img width="601" height="530" alt="Screenshot 2025-11-03 232307" src="https://github.com/user-attachments/assets/55dd417d-6acc-424f-8898-eea2229cc19e" />

**Key Observation**: The validation accuracy (orange line) is consistently higher than the training accuracy (blue line). This is a classic and positive sign of Dropout working correctly. The model is "nerfed" during training (as neurons are randomly dropped) but uses its full power during validation, resulting in better performance. This shows the model is generalizing well and is not overfitting.

### CNN Confusion Matrix

(This is your new confusion matrix)

**Analysis**: The CNN's performance is a clear success. The errors that plagued the ANN have been almost completely solved:

**'9' vs. '4' Errors**: Dropped from 12 to 4.

**'9' vs. '7' Errors**: Dropped from 8 to 2.

The model's primary remaining confusion (e.g., '2' vs. '7', with 11 and 13 errors) is a much harder, well-known problem, but the original spatial issues were fixed.

## Conclusion

The CNN (98.71%) clearly outperformed the ANN (97.89%). By using Conv2D layers to analyze the 2D image structure before flattening, the model was able to learn robust spatial features (like loops vs. open lines), which directly solved the ANN's main failure points.
