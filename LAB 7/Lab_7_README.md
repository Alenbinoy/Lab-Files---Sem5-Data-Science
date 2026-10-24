# Lab 7: Keras MLP for Multiclass Classification (Dry Beans)

Notebook: `Lab_7.ipynb`
Course: Artificial Neural Networks and Deep Learning

---

## 1. What are we trying to do?

Tell 7 types of dry bean apart using 16 measurements taken from photos of each bean (area, perimeter, axis lengths, shape ratios). This is **multiclass classification**: one input, one of seven labels.

The lab compares three activation functions (ReLU, Sigmoid, Tanh) and three optimisers (Adam, SGD, RMSprop).

## 2. The dataset

Dry Bean Dataset (UCI / Kaggle): 13,611 beans, 16 features, 7 classes (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira). No missing values.

The classes are not balanced. Dermason has 3,546 samples and Bombay only 522, a ratio of about 6.8 to 1. Several features are almost the same number measured differently (for example ConvexArea and EquivDiameter correlate at 0.985).

Preprocessing: labels are encoded and one-hot encoded, the split is 80% train and 20% test with `stratify` so every class keeps its share, and features are standardised using the training set only (so nothing leaks from the test set).

## 3. The model

```
Input (16) -> Dense 128 -> Dense 64 -> Dense 7 (softmax)
```

Softmax turns the 7 outputs into probabilities that sum to 1. The loss is categorical cross-entropy, which punishes confident wrong answers hardest. 60 epochs, batch size 64, learning rate 0.001, 20% of training data held out for validation.

## 4. Results (test set)

| Model | Activation | Optimiser | Accuracy | F1 |
|---|---|---|---|---|
| 1 | ReLU | Adam | 0.9251 | 0.9249 |
| 2 | Sigmoid | Adam | 0.9207 | 0.9206 |
| 3 | Tanh | Adam | 0.9232 | 0.9231 |
| 4 | ReLU | SGD (momentum 0.9) | 0.9236 | 0.9238 |
| 5 | ReLU | RMSprop | 0.9258 | 0.9256 |

Final model (ReLU + Adam, retrained and evaluated once more): **92.4% accuracy, F1 0.924**.

## 5. What the numbers say

- **All five models land between 92.1% and 92.6%.** The gaps are tiny and within run to run noise, so the honest summary is that activation and optimiser barely matter on this dataset. The features are already informative and the network is small.
- **ReLU led on activation**, but only by a few hundredths of a point over Tanh and Sigmoid.
- **Optimisers differ in speed more than in the final score.** Adam and RMSprop passed 90% validation accuracy in the first epoch. SGD took 6 epochs.
- **The weak spot is Sira.** It has the lowest per-class F1 (about 0.87). Sira, Dermason and Horoz have similar shapes, so they get confused with each other most.
- **No sign of overfitting.** Training and validation curves stay close together.

## 6. Small inconsistencies in the notebook

- The auto-generated answers name RMSprop as the best optimiser, then name ReLU + Adam as the best combination. The table shows RMSprop barely ahead of the rest.
- ReLU + Adam appears with slightly different scores in different runs (0.9251, 0.9236, 0.9240) because training is not exactly repeatable. That is the noise mentioned above.
- The comparison table labels the models 1 to 5 while the training cells call them 1 to 6.

## 7. How to run

Python 3, TensorFlow 2.x, scikit-learn, pandas, matplotlib, seaborn. The dataset comes from Kaggle or the UCI mirror. Open `Lab_7.ipynb` in Colab and run all cells in order.
