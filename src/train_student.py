import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
# import tokenizer
from dataset_loader import DistillationDataset
from datasets import load_dataset


# --- HYPERPARAMETERS ---
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
EPOCHS = 10
TEMPERATURE = 2.0  # Softens the probability distribution
ALPHA = 1.0        # Weight for Distillation Loss
VOCAB_SIZE = 4672  # Standard Chess Move Vocab
SEQ_LEN = 77       # Input sequence length

# --- STUDENT MODEL ARCHITECTURE ---
class StudentChessTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        # Add positional encoding
        x = self.embedding(x) + self.pos_embedding[:, :T, :]
        x = self.transformer(x)
        # Predict logits for the LAST token (next move prediction)
        logits = self.fc_out(x) 
        return logits

# --- LOSS FUNCTION ---
def distillation_loss(student_logits, teacher_logits, temp):
    """
    KL Divergence Loss between Student and Teacher soft targets.
    """
    # LogSoftmax for Student
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    
    # Softmax for Teacher (Teacher logits are raw, so we apply softmax)
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
    
    # KLDivLoss expects input in log-space and target in prob-space
    # reduction='batchmean' is mathematically proper for KL
    loss = nn.KLDivLoss(reduction="batchmean")(student_log_probs, teacher_probs)
    
    # Scale loss by T^2 as per Hinton's Distillation paper
    return loss * (temp ** 2)

def create_collate_fn(teacher):
    def collate_fn(batch):
        tokens = torch.stack([item['tokens'] for item in batch])
        teacher_logits = torch.stack([item['teacher_logits'] for item in batch])
        return tokens, teacher_logits
    return collate_fn
# --- TRAINING LOOP ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Load Data
    # Ensure you ran the JAX exporter first!
    # 1. Enable streaming
    ds = load_dataset(
        "Lichess/chess-position-evaluations", 
        split="train[0:60_000_000]", # Download the first 60M samples
    )

# 2. Filter for columns 'fen' and 'cp'
    ds = ds.select_columns(["fen", "cp", 'mate'])
    split_dataset = ds.train_test_split(test_size=0.2, seed=42)

    train_data, val_data = split_dataset['train'], split_dataset['test']
    exit(0)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Initialize Student
    student = StudentChessTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=256,  # Significantly smaller than Teacher (usually 512+)
        num_layers=6,   # Fewer layers (Teacher likely has 12+)
        num_heads=8
    ).to(device)
    
    optimizer = optim.AdamW(student.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 3. Training
    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for tokens, teacher_logits in loop:
            tokens, teacher_logits = tokens.to(device), teacher_logits.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            student_logits = student(tokens)
            
            loss = distillation_loss(student_logits, teacher_logits, TEMPERATURE)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # Validation
        student.eval()
        val_loss = 0
        with torch.no_grad():
            for tokens, teacher_logits in val_loader:
                tokens, teacher_logits = tokens.to(device), teacher_logits.to(device)
                student_logits = student(tokens)
                loss = distillation_loss(student_logits, teacher_logits, TEMPERATURE)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1} Val Loss: {val_loss / len(val_loader):.4f}")
        
        # Save Checkpoint
        torch.save(student.state_dict(), f"student_chess_model_ep{epoch+1}.pt")

if __name__ == "__main__":
    main()