import torch 
import torch.nn as nn

class VeryManualLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_x = nn.Parameter(torch.randn(input_size, 4 * hidden_size) * 0.1)
        self.W_h = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) * 0.1)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))

    #! h_prev is the previous hidden, short-term, c_prev is the previous cell state, long-term 
    def forward(self, x_t, h_prev, c_prev):
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.b
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=-1)

        i_t = torch.sigmoid(i_t) #* input gate (green in the diagram)
        f_t = torch.sigmoid(f_t) #* forget gate (blue in the diagram)
        g_t = torch.tanh(g_t)    #* candidate (orange in the diagram)
        o_t = torch.sigmoid(o_t) #* output gate (purple in the diagram)

        c_t = f_t * c_prev + i_t * g_t 
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class LSTMFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.cell = VeryManualLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, initial_state=None):
        """
        x: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.shape

        if initial_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = initial_state

        hidden_states = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, h_t, c_t)
            hidden_states.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_states, dim=1)   # (batch, seq_len, hidden)
        y = self.fc(hidden_seq[:, -1, :])              # use final hidden state

        return y, (h_t, c_t)

# Example usage
torch.manual_seed(0)

batch_size = 4
seq_len = 10
input_size = 3
hidden_size = 8
output_size = 1

x = torch.randn(batch_size, seq_len, input_size)
y_true = torch.randn(batch_size, output_size)

model = LSTMFromScratch(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    optimizer.zero_grad()

    y_pred, (h_n, c_n) = model(x)
    loss = criterion(y_pred, y_true)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    lstm = VeryManualLSTMCell(input_size=3, hidden_size=5)
    print(f"self.W_x: {lstm.W_x.shape}")
    print(f"self.W_x type: {type(lstm.W_x)}")