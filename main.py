import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# 1. Toy Support Ticket Dataset: (text, label_id)
# Labels: 0 = Technical Bug, 1 = Billing, 2 = Account Access
raw_data = [
    ("app crashes when clicking save button", 0),
    ("error 500 on dashboard page", 0),
    ("charged twice on my credit card invoice", 1),
    ("need refund for annual subscription", 1),
    ("forgot my password cannot login", 2),
    ("two factor authentication code not arriving", 2),
]

# 2. Build a simple Vocabulary mapping words -> integer IDs
# Special tokens: <PAD>=0 (for padding short sentences), <UNK>=1 (for unknown words)
vocab = {"<PAD>": 0, "<UNK>": 1}
for text, _ in raw_data:
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)

def encode_text(text: str, vocab: dict) -> list[int]:
    """Converts a string into a list of integer token IDs."""
    return [vocab.get(word, vocab["<UNK>"]) for word in text.lower().split()]

print(f"Vocab size: {len(vocab)}")
print(f"Encoded sample: {encode_text('app crashes on login', vocab)}")


class TicketDataset(Dataset):
    def __init__(self, data: list[tuple[str, int]], vocab: dict):
        self.data = data
        self.vocab = vocab

    def __len__(self) -> int:
        # Returns total number of samples
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Fetches ONE sample at index `idx` and converts to Tensors
        text, label = self.data[idx]
        encoded_ids = encode_text(text, self.vocab)
        
        # Token IDs and Classification Labels MUST be torch.long (int64)
        return torch.tensor(encoded_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """
    `batch` is a list of tuples returned by Dataset.__getitem__:
    [(text_tensor_1, label_1), (text_tensor_2, label_2), ...]
    """
    texts, labels = zip(*batch)
    
    # Pad variable-length text tensors with 0 (<PAD>)
    # batch_first=True makes the output shape: (batch_size, max_seq_len)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    
    # Stack scalar labels into a 1D tensor of shape: (batch_size,)
    labels_tensor = torch.stack(labels)
    
    return padded_texts, labels_tensor


#TODO: Instantiate  dataset = TicketDataset(raw_data, vocab)
# Wrap it in a loader = DataLoader(...) with batch_size=2, shuffle=True, and collate_fn=collate_batch.
# Loop over loader and print out each batch's padded_texts tensor, its .shape, and the labels tensor.
# Bonus Question: Look at the padded batch output. Why is it better to pad dynamically per batch inside collate_batch rather than padding the entire dataset upfront to the single longest document in the whole dataset?
"""Answer: Because text can be in different length. Text in dataset can be way longer, that in batch. With dynamic option we will have less zeros """
dataset = TicketDataset(raw_data, vocab)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)
for padded_texts, labels in loader:
    print("Texts: ")
    print(padded_texts)
    print("Shape: ")
    print(padded_texts.shape)
    print("Labels: ", labels)
    print("-------------")


class TicketClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        # 1. Lookup table: maps integer ID -> dense vector of size `embed_dim`
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Classifier head: maps the averaged sentence vector -> 3 class logits
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len) -> e.g., (2, 6)
        
        # Step 1: Look up embeddings for each token ID
        # embedded shape: (batch_size, seq_len, embed_dim) -> e.g., (2, 6, 16)
        embedded = self.embedding(x)
        
        # Step 2: Average the word vectors across the sequence dimension (dim=1)
        # Note: To do true mean pooling ignoring <PAD> zeros, we divide by true lengths:
        mask = (x != 0).unsqueeze(-1).float()                  # (batch_size, seq_len, 1)
        sentence_vector = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # sentence_vector shape: (batch_size, embed_dim) -> e.g., (2, 16)
        
        # Step 3: Pass through linear layer to get raw scores (logits) for each class
        # logits shape: (batch_size, num_classes) -> e.g., (2, 3)
        logits = self.fc(sentence_vector)
        return logits


# 1. Instantiate Model, Loss, and Optimizer
model = TicketClassifier(vocab_size=len(vocab), embed_dim=16, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# 2. Training Loop (Run for 30 epochs over your DataLoader)
model.train()  # Put model in training mode
for epoch in range(30):
    total_loss = 0.0
    for padded_texts, labels in loader:
        # TODO 1: Clear old gradients using optimizer
        # TODO 2: Forward pass -> get logits from model(padded_texts)
        # TODO 3: Compute loss using criterion(logits, labels)
        # TODO 4: Backward pass -> compute gradients
        # TODO 5: Update weights using optimizer
        
        total_loss += loss.item()
        
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss / len(loader):.4f}")

