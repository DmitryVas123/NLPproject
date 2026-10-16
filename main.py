import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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

dataset = TicketDataset(raw_data, vocab)
loader = DataLoader(batch_size=2, shuffle=True, collate_fn=collate_batch)