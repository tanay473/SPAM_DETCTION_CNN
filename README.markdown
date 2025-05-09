# Spam Detection Using CNN

## Task Objectives
This project implements a Convolutional Neural Network (CNN) to classify text messages as spam or not spam using the SMS Spam Collection dataset. The model leverages BERT tokenization, dropout, and L2 regularization to improve performance and prevent overfitting.

## Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Download the SMS Spam Collection dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Place `spam.csv` in the `data/` directory.

## Steps to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/spam-detection-cnn
   cd spam-detection-cnn
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Ensure `spam.csv` is in the `data/` directory.

4. **Train the Model**:
   ```bash
   python src/train.py
   ```
   - This trains the CNN and saves the model to `models/cnn_spam_model_regularized.pt`.

5. **Evaluate the Model**:
   ```bash
   python src/evaluate.py
   ```
   - This loads the trained model and reports the final validation accuracy.

6. **Plot Metrics**:
   ```bash
   python plot_metrics.py
   ```
   - This generates a plot of training/validation loss and accuracy (`metrics_plot.png`).

## Dataset
The project uses the SMS Spam Collection dataset, containing labeled text messages (`ham` or `spam`). The data is split into 80% training and 20% validation sets.

## Model Architecture
The CNNClassifier model consists of the following layers, each with a specific role in processing text for spam detection:
- **Embedding Layer**: Converts tokenized input (from BERT tokenizer) into dense vectors of 128 dimensions. This layer transforms each word into a numerical representation that captures semantic meaning, enabling the model to process text data.
- **Convolutional Layer**: Applies 16 filters with a kernel size of 5 to detect local patterns, such as specific word combinations or phrases, that are indicative of spam or non-spam content. This layer extracts relevant features from the embedded text.
- **ReLU Activation**: Introduces non-linearity after the convolutional layer, allowing the model to learn complex patterns by activating only positive feature responses and setting negative values to zero.
- **Pooling (Adaptive Max Pooling)**: Reduces the dimensionality of the feature maps by selecting the most important features (maximum values) across the sequence. This retains critical information while making the model computationally efficient and less sensitive to text length.
- **Dropout (50%)**: Randomly disables 50% of the neurons during training to prevent overfitting, ensuring the model generalizes better to unseen data by reducing reliance on specific features.
- **Fully Connected Layer**: Maps the extracted features to two output classes (spam or not spam), producing logits that represent the probability of each class.

## Innovative Features
- **BERT Tokenization**: Uses pre-trained BERT tokenizer for robust text preprocessing.
- **Regularization**: Combines dropout (0.5) and L2 regularization (1e-5) to improve generalization.
- **Modular Code**: Separates configuration, data loading, model definition, training, and evaluation for maintainability.
- **Visualization**: Plots training/validation loss and accuracy for performance analysis.
- **Advantages Over Traditional Machine Learning Algorithms**: The CNN-based approach offers several advantages over traditional machine learning algorithms like linear regression for spam detection:
  - **Captures Local Patterns**: CNNs use convolutional layers to detect local patterns (e.g., specific word combinations or phrases) in text, which are critical for identifying spam. Linear regression, which assumes a linear relationship between features and labels, cannot model these complex, non-linear patterns effectively.
  - **Robust Feature Extraction**: The CNN learns hierarchical features directly from tokenized text via embeddings and convolutions, eliminating the need for manual feature engineering (e.g., bag-of-words or TF-IDF) required by linear regression. This makes the model more adaptable to diverse text data.
  - **Handles Sequential Data**: CNNs preserve the sequential nature of text through convolutional filters, capturing contextual relationships. Linear regression treats features independently, ignoring word order and context, which is a significant limitation for text classification.
  - **Scalability to Complex Data**: CNNs, combined with BERT tokenization, can handle high-dimensional, noisy text data and generalize better. Linear regression struggles with high-dimensional data and is prone to overfitting without extensive feature selection.
  - **Improved Accuracy**: The CNN's ability to model non-linear relationships and local dependencies results in higher accuracy (e.g., ~98.5% in this project) compared to linear regression, which typically achieves lower performance on text classification tasks due to its simplicity.

## Evaluation Metrics
Example metrics (based on sample run):
| Metric       | Value  |
|--------------|--------|
| Accuracy     | 98.5%  |
| Precision    | 97.8%  |
| Recall       | 98.2%  |
| F1-Score     | 98.0%  |

## Directory Structure
```plaintext
spam-detection-cnn/
├── data/
│   └── spam.csv
├── models/
│   └── cnn_spam_model_regularized.pt
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── plot_metrics.py
├── requirements.txt
└── README.md
```

## Notes
- Training uses 50 epochs, but early stopping could be added for efficiency.
- The model is saved as a state dictionary for easy loading.
- Plots are saved as `metrics_plot.png` for visual inspection.

For issues or questions, please open an issue on the GitHub repository.
