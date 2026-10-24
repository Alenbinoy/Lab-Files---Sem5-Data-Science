# Lab 8: RNN Text Classification on IMDB Reviews (Keras SimpleRNN)

Notebook: `LAB_8.ipynb`
Course: Artificial Neural Networks and Deep Learning

---

## 1. What are we trying to do?

Teach a computer to read a movie review and decide whether the writer liked the film (**Positive**) or not (**Negative**).

This is **binary sentiment classification** on text. Earlier labs used numbers (Iris measurements) and images (Horses vs Humans). Here the input is a **sequence of words**, and the order of the words matters ("not good" means something different from "good"). That is why we use a **Recurrent Neural Network (RNN)**, a network that reads one word at a time and keeps a running memory of what it has read so far.

The lab also studies how three hyperparameters (RNN size, number of epochs, batch size) change the result, and ends with a challenge: stacking two RNN layers.

---

## 2. The dataset

**IMDB movie reviews**, loaded with `keras.datasets.imdb`.

| Item | Value |
|---|---|
| Training reviews | 25,000 (12,500 positive, 12,500 negative) |
| Test reviews | 25,000 (12,500 positive, 12,500 negative) |
| Review length | min 11, max 2,494, mean 238.7, median 178 words |
| Vocabulary kept | the 10,000 most frequent words |

Each review is already stored as a list of integers, where each integer is a word. Rarer words outside the top 10,000 are replaced by a single "unknown word" index (2). The classes are perfectly balanced, so plain accuracy is a fair metric.

---

## 3. Preprocessing

Neural networks need inputs of the same size, but reviews have very different lengths. So every review is **padded or truncated to exactly 200 tokens**:

- shorter reviews get zeros added at the front (`padding='pre'`)
- longer reviews keep only their last 200 tokens (`truncating='pre'`)

After this, `x_train` and `x_test` both have shape `(25000, 200)`.

---

## 4. The model

```
Input (200 word indices)
    -> Embedding (10,000 words -> 128 numbers each)
    -> SimpleRNN (64 units)
    -> Dense 1 neuron (Sigmoid) -> P(positive)
```

- **Embedding**: turns each word index into a vector of 128 learned numbers, so that words with similar meaning can end up with similar vectors. It holds 1,280,000 of the 1,292,417 parameters.
- **SimpleRNN**: reads the 200 vectors in order, updating a 64-number memory after each word. Its final memory summarises the whole review.
- **Dense + Sigmoid**: turns that summary into one probability between 0 and 1. 0.5 or above is called Positive.

Training setup: Adam optimiser, binary cross-entropy loss, batch size 64, 5 epochs, 20% of the training data held out as validation. All seeds are fixed to 42 so runs can be repeated.

---

## 5. Baseline results

| Measure | Value |
|---|---|
| Final training accuracy | 87.26% |
| Final validation accuracy | 71.26% |
| Test accuracy | **71.10%** |
| Test loss | 0.7196 |

Test accuracy is lower than training accuracy by about 16 points. That gap is **overfitting**: the model partly memorises the training reviews instead of learning something general.

The training curves are also jagged. Validation accuracy peaks at 80.1% after epoch 1, falls to 58.7% at epoch 3, then partly recovers. This is typical of a SimpleRNN, which struggles to learn from information 200 steps back (the **vanishing and exploding gradient** problem). The best model here would arguably be the one after only 1 or 2 epochs.

---

## 6. Hyperparameter experiments

One setting was changed at a time and everything else stayed at the baseline (64 units, 5 epochs, batch 64). The baseline run is reused for A2, B2 and C2.

| ID | Setting | Train acc | Val acc | Test acc | Time |
|---|---|---|---|---|---|
| A1 | RNN = 32 | 0.9886 | 0.7932 | **0.7848** | 31 s |
| A2 | RNN = 64 | 0.8726 | 0.7126 | 0.7110 | 41 s |
| A3 | RNN = 128 | 0.8116 | 0.7074 | 0.7070 | 86 s |
| B1 | Epochs = 3 | 0.9067 | 0.7632 | 0.7658 | 28 s |
| B2 | Epochs = 5 | 0.8726 | 0.7126 | 0.7110 | 41 s |
| B3 | Epochs = 10 | 0.9729 | 0.7008 | 0.6999 | 90 s |
| C1 | Batch = 32 | 0.8966 | 0.7322 | 0.7457 | 59 s |
| C2 | Batch = 64 | 0.8726 | 0.7126 | 0.7110 | 41 s |
| C3 | Batch = 128 | 0.9744 | 0.7852 | 0.7821 | 52 s |

What this shows:

- **RNN units:** a bigger layer did not help. 32 units scored best, and 128 units was the slowest and the worst. More parameters means more room to overfit and a harder network to train.
- **Epochs:** more training made things worse. Training accuracy was highest at 10 epochs while test accuracy was lowest. Fewer epochs (or early stopping) is better.
- **Batch size:** 128 scored best, then 32, then 64. That order is not smooth, which tells us the runs are noisy.

SimpleRNN training is unstable, so treat gaps of a few points as trends, not exact effects. **Best setups seen:** 32 units (78.48%) and batch size 128 (78.21%).

Note: the timings in the notebook's written explanation differ slightly from the printed run times above. The table here uses the printed outputs.

---

## 7. Predicting on new text

`predict_review(text)` turns raw text into the same integer encoding the dataset uses (word index plus 3, because indices 0 to 3 are reserved for padding, start and unknown), pads it to 200 and runs the model.

Five hand-written reviews were all classified correctly:

| Review | Expected | Probability positive |
|---|---|---|
| 1 | Positive | 0.97 |
| 2 | Negative | 0.04 |
| 3 | Positive | 0.84 |
| 4 | Negative | 0.01 |
| 5 | Positive | 0.98 |

These are easy cases with strongly worded reviews ("wonderful", "terrible", "boring"). With about 71% test accuracy, the model would be wrong more often on sarcasm, mixed opinions or negation.

---

## 8. Challenge: a two-layer RNN

```
Embedding -> SimpleRNN(64, return_sequences=True) -> SimpleRNN(64) -> Dense(sigmoid)
```

`return_sequences=True` makes an RNN output its memory at **every** word (shape `(batch, 200, 64)`) instead of only at the last word (shape `(batch, 64)`). The second RNN needs a sequence as input, so every RNN layer except the last must set it.

| | Single-layer | Two-layer |
|---|---|---|
| Test accuracy | 71.10% | 69.50% |
| Training accuracy | 87.26% | 96.95% |
| Validation accuracy | 71.26% | 70.04% |
| Test loss | 0.7196 | 0.9862 |
| Training time | 41 s | 73 s |

The deeper model fits the training data much better but does slightly worse on unseen reviews, and takes almost twice as long. So **another layer does not always help**. It only pays off when the task needs the extra depth and the model is regularised and trained stably.

---

## 9. Conclusion

A SimpleRNN reaches about 71% test accuracy on IMDB sentiment, with clear overfitting and unstable training. Smaller layers (32 units), fewer epochs and a larger batch size (128) gave better results than the defaults, and a second RNN layer did not help. The main limits are the vanishing gradient over 200 steps, strong overfitting, and results that change from run to run. **LSTM or GRU** layers (which use gates to keep long-range information), plus dropout and early stopping, are the natural next step.

---

## 10. How to run

Requirements: Python 3, TensorFlow 2.x, NumPy, pandas, matplotlib.

1. Open `LAB_8.ipynb` in Jupyter or VS Code.
2. Run all cells in order. The IMDB data downloads automatically on first run.
3. The full notebook, including all nine experiments, takes roughly 10 to 15 minutes on a CPU.
