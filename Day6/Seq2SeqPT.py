import torch
import torch.nn as nn
import torch.optim as optim
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedding = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)

        embedding = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))

        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        x = target[:, 0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[:, t] = output

            best_guess = output.argmax(1)

            x = target[:, t] if random.random() < teacher_forcing_ratio else best_guess

        return outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 10
output_size = 10
embed_size = 32
hidden_size = 64
num_layers = 2

encoder = Encoder(input_size, embed_size, hidden_size, num_layers).to(device)
decoder = Decoder(output_size, embed_size, hidden_size, num_layers).to(device)

model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

source = torch.tensor([[1,2,3,4]]).to(device)
target = torch.tensor([[0,2,3,4,5]]).to(device)

for epoch in range(200):
    output = model(source, target)

    output = output[:, 1:].reshape(-1, output_size)
    target_reshaped = target[:, 1:].reshape(-1)

    loss = criterion(output, target_reshaped)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
print("Training complete!")

print("Testing on new data...")
test_source = torch.tensor([[5,6,7,8]]).to(device)
test_target = torch.tensor([[0,6,7,8,9]]).to(device)
test_output = model(test_source, test_target, teacher_forcing_ratio=0)
test_output = test_output[:, 1:].argmax(2)
print("Predicted:", test_output.cpu().numpy())
print("Expected:", test_target[:, 1:].cpu().numpy())