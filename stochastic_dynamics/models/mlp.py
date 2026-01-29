import torch
import torch.nn as nn


class ARMLP(nn.Module):
    def __init__(self, seq_len=600, n_classes=5, max_ar_order=6, hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.max_ar_order = max_ar_order
        
        # Simple MLP: input -> hidden -> hidden -> coefficients
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * max_ar_order)
        )
    
    def forward(self, x, temperature=1.0):
        N = x.shape[0]
        device = x.device
        
        # MLP outputs all coefficients directly
        coeffs = self.mlp(x).view(N, self.seq_len, self.max_ar_order)  # (N, 600, 6)
        
        # No order prediction — return zeros for compatibility
        p_logits = torch.zeros(N, self.n_classes, device=device)
        p_hard = torch.zeros(N, dtype=torch.long, device=device)
        
        # AR reconstruction: x_hat[t] = sum_k coeffs[t,k] * x[t-k-1]
        x_lagged = torch.zeros(N, self.seq_len, self.max_ar_order, device=device)
        for k in range(self.max_ar_order):
            x_lagged[:, k+1:, k] = x[:, :self.seq_len - k - 1]
        
        x_hat = (coeffs * x_lagged).sum(dim=-1)  # (N, 600)
        
        return coeffs, p_logits, p_hard, x_hat