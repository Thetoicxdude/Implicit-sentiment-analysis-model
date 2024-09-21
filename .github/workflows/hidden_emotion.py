import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext.data.utils import get_tokenizer
import random
import math
import os
import json
import nltk
from nltk.corpus import wordnet as wn


config = {
    "split_ratio": [0.8, 0.2],
    "batch_size": 64,
    "seed": 42,
    "num_epochs": 100,
    "num_models": 5,
    "patience": 5  
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

random_seed = 42
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

nltk.download('wordnet')


tokenizer = get_tokenizer("basic_english")


def synonym_replacement(words, n=1):
    """Perform synonym replacement for the given words."""
    new_words = words.copy()
    for _ in range(n):
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if isinstance(word, str):  
            synonyms = get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words[idx] = synonym
    return new_words

def get_synonyms(word):
    """Retrieve synonyms for a given word using WordNet."""
    synonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms


TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False, dtype=torch.int64)


EMOTION_LABELS = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
HIDE_EMOTION_LABELS = ["no emotion", "anger", "Care", "fear", "anxiety", "sadness", "craving", "anticipation", "pleasure", "satisfaction"]


try:
    with open("AI_daliydia_test.json", "r", encoding="utf-8") as file:
        train_data = json.load(file)

    with open("AI_daliydia_test2.json", "r", encoding="utf-8") as file:
        test_data = json.load(file)

    train_dialogs = [row["row"]["dialog"] for row in train_data["rows"]]
    train_hide_emotion = [row["row"]["hide_emotion"] for row in train_data["rows"]]
    train_emotions = [row["row"]["emotion"] for row in train_data["rows"]]

    test_dialogs = [row["row"]["dialog"] for row in test_data["rows"]]
    test_hide_emotion = [row["row"]["hide_emotion"] for row in test_data["rows"]]
    test_emotions = [row["row"]["emotion"] for row in test_data["rows"]]

    print("Number of rows in the train data:", len(train_data["rows"]))
    print("Number of rows in the test data:", len(test_data["rows"]))
except Exception as e:
    print(f"Error loading or processing JSON data: {e}")


fields = [("text", TEXT), ("hide_emotion", LABEL), ("emotion", LABEL)]


train_examples = [torchtext.data.Example.fromlist([train_dialogs[i], train_hide_emotion[i], train_emotions[i]], fields)
                  for i in range(len(train_dialogs))]
test_examples = [torchtext.data.Example.fromlist([test_dialogs[i], test_hide_emotion[i], test_emotions[i]], fields)
                 for i in range(len(test_dialogs))]


train_dataset = torchtext.data.Dataset(train_examples, fields)
test_dataset = torchtext.data.Dataset(test_examples, fields)


train_data, val_data = train_dataset.split(split_ratio=config["split_ratio"])

train_iterator, val_iterator, test_iterator = torchtext.data.BucketIterator.splits(
    (train_data, val_data, test_dataset),
    batch_size=config["batch_size"],
    sort_key=lambda x: len(x.text),
    sort_within_batch=False
)


TEXT.build_vocab(train_dataset, vectors="glove.6B.100d")

# Define sentiment analysis model
class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embedding=None,
                 num_layers=1, bidirectional=False, dropout=0.5):
        super(SentimentAnalysisModel, self).__init__()

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional,
                           batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, text, text_lengths):
        embedded_text = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded_text, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.layer_norm(output)
        output = self.dropout(output)
        output = self.fc1(output[:, -1, :])  
        output = F.relu(output)
        output = self.fc2(output)
        return output


output_dim = len(HIDE_EMOTION_LABELS)
models = []
for _ in range(config["num_models"]):
    model = SentimentAnalysisModel(vocab_size=len(TEXT.vocab),
                                   embedding_dim=100,
                                   hidden_dim=256,
                                   output_dim=output_dim,
                                   bidirectional=True)
    models.append(model.to(device))


criterion = nn.CrossEntropyLoss()
optimizers = [optim.Adam(model.parameters(), weight_decay=1e-5) for model in models]

best_val_loss = float('inf')
epochs_no_improve = 0


def train(model, iterator, optimizer, criterion, device, augmentations=True):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        text, text_lengths = batch.text
        surface_emotion = batch.hide_emotion.long()

        if augmentations:
            augmented_texts = [synonym_replacement(text[i].tolist()) for i in range(len(text))]
            augmented_texts = [torch.LongTensor(aug_text).to(device) for aug_text in augmented_texts]
            augmented_texts = pad_sequence(augmented_texts, batch_first=True)

            text = augmented_texts

        optimizer.zero_grad()

        output = model(text, text_lengths)
        loss = criterion(output.view(-1, output.shape[-1]), batch.hide_emotion.view(-1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            surface_emotion = batch.emotion.long()

            output = model(text, text_lengths)
            loss = criterion(output.view(-1, output.shape[-1]), batch.hide_emotion.view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Training and validation loop with early stopping
for epoch in range(config["num_epochs"]):
    for i, model in enumerate(models):
        train_loss = train(model, train_iterator, optimizers[i], criterion, device)
        val_loss = evaluate(model, val_iterator, criterion)

        print(f'Epoch: {epoch + 1:02} | Model: {i + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_model_{i}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

# Load the best model and test performance
for i, model in enumerate(models):
    if os.path.exists(f"best_model_{i}.pt"):
        model.load_state_dict(torch.load(f"best_model_{i}.pt"))
        test_loss = evaluate(model, test_iterator, criterion)
        print(f'Model: {i + 1:02} | Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
