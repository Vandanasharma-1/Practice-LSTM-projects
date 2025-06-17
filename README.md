# Tweet Sentiment Analysis with Bidirectional LSTM

## Project Overview

This repository contains a comprehensive deep learning project focused on classifying the sentiment of tweets (as Positive or Negative). The project demonstrates an end-to-end Machine Learning pipeline, from data preprocessing and tokenization to building, training, and evaluating a Bidirectional Long Short-Term Memory (LSTM) neural network.

The goal was to build a highly accurate and reliable sentiment classifier capable of generalizing well to unseen social media data, a critical capability for applications like brand monitoring, customer feedback analysis, and trend prediction.

## Key Features

* **Data Preprocessing:** Robust handling of raw tweet data, including cleaning (removing URLs, mentions, hashtags, punctuation, numbers, extra spaces), and tokenization, with careful consideration for retaining crucial sentiment-altering words (e.g., negations).
* **Text Vectorization:** Utilizes Keras's `Tokenizer` for word-to-integer mapping and `pad_sequences` to ensure uniform input length for the LSTM model.
* **Bidirectional LSTM Architecture:** Implements a powerful Bidirectional LSTM network, capable of capturing dependencies in text sequences from both forward and backward directions, enhancing contextual understanding.
* **Model Training & Optimization:**
    * **Early Stopping:** Prevents overfitting by monitoring validation loss and stopping training when improvement plateaus.
    * **Model Checkpointing:** Saves the best performing model weights based on validation loss, ensuring the optimal model version is retained.
* **Comprehensive Evaluation:** Rigorous evaluation using standard metrics such as accuracy, precision, recall, F1-score, and a confusion matrix to provide a detailed understanding of model performance.
* **Visualization:** Includes plots for training/validation loss and accuracy over epochs, visually demonstrating the model's learning curve and generalization behavior.

## Project Structure

* `LSTM.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model definition, training, evaluation, and visualization.
* `best_sentiment_model.keras`: (This file will be generated after training) The saved weights of the best-performing model identified by `ModelCheckpoint`.
* `(sentiment_tweets3.csv)`

## Technical Stack

* **Python 3.10.16**
* **TensorFlow / Keras:** For building and training the deep learning model.
* **Numpy:** For numerical operations and data handling.
* **Pandas:** For data loading and manipulation.
* **Scikit-learn:** For data splitting (`train_test_split`) and performance metrics (`classification_report`, `confusion_matrix`).
* **NLTK:** For text preprocessing utilities (e.g., stopwords, WordNetLemmatizer).
* **Matplotlib / Seaborn:** For data visualization.

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Vandanasharma-1/Practice-LSTM-projects.git](https://github.com/Vandanasharma-1/Practice-LSTM-projects.git)
    cd Practice-LSTM-projects
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install the required libraries:**
    ```bash
    pip install tensorflow pandas numpy scikit-learn nltk matplotlib seaborn
    ```
4.  **Download NLTK data:**
    Open a Python interpreter or add the following lines at the beginning of your notebook to download necessary NLTK components:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    ```
5.  **Place your dataset:** Ensure your tweet dataset (e.g., `your_tweet_data.csv`) is placed in the project root directory or update the path in the notebook accordingly.
6.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open `LSTM.ipynb` and run all cells.



This strong performance underscores the model's ability to accurately differentiate between positive and negative tweet sentiments.

## Contact

Feel free to connect with me on LinkedIn if you have any questions or would like to discuss this project further!

[LinkedIn Profile Link Here - e.g., `https://www.linkedin.com/in/vandana-sharma-20a5ab24a/`]

---
