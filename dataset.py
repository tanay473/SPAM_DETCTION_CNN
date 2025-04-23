import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from config import MAX_LEN, BATCH_SIZE

class SpamDataset(Dataset):
    """Custom Dataset for Spam classification."""
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.texts[idx],
                                   truncation=True,
                                   padding='max_length',
                                   max_length=self.max_len,
                                   return_tensors="pt")
        input_ids = tokenized['input_ids'].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label

def load_data(file_path="spam.csv"):
    """Load and preprocess the spam dataset."""
    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")[['v1', 'v2']]
        df.columns = ['label', 'text']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        texts, labels = df['text'].tolist(), df['label'].tolist()
        print(f"Loaded {len(texts)} messages.")
    except FileNotFoundError:
        print("Error: spam.csv not found. Please ensure the file is in the correct directory.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading or processing spam.csv: {e}")
        exit()

    return texts, labels

def get_dataloaders(texts, labels):
    """Split data and create DataLoaders."""
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
    val_dataset = SpamDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, tokenizer