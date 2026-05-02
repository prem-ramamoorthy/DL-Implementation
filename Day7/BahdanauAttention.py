import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()
        self.W1 = nn.Linear(enc_hidden_dim, attn_dim)
        self.W2 = nn.Linear(dec_hidden_dim, attn_dim)
        self.V = nn.Linear(attn_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):

        decoder_hidden = decoder_hidden.unsqueeze(1)

        score = self.V(torch.tanh(
            self.W1(encoder_outputs) + self.W2(decoder_hidden)
        ))

        attn_weights = F.softmax(score, dim=1)

        context = attn_weights * encoder_outputs
        context = context.sum(dim=1)

        return context, attn_weights

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)

        outputs = outputs[:, :, :hidden.size(2)] + outputs[:, :, hidden.size(2):]

        hidden = hidden[0] + hidden[1]
        hidden = hidden.unsqueeze(0)

        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim, attn_dim)

        self.rnn = nn.GRU(enc_hidden_dim + emb_dim, dec_hidden_dim, batch_first=True)
        self.fc = nn.Linear(dec_hidden_dim, output_dim)

    def forward(self, input_token, hidden, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)

        context, attn_weights = self.attention(
            encoder_outputs, hidden.squeeze(0)
        )

        context = context.unsqueeze(1)

        rnn_input = torch.cat((embedded, context), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        prediction = self.fc(output.squeeze(1))

        return prediction, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)

        encoder_outputs, hidden = self.encoder(src)

        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(
                input_token, hidden, encoder_outputs
            )

            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = trg[:, t] if teacher_force else top1

        return outputs

INPUT_DIM = OUTPUT_DIM = 10
EMB_DIM = 16
HIDDEN_DIM = 32
ATTN_DIM = 32

encoder = Encoder(INPUT_DIM, EMB_DIM, HIDDEN_DIM).to(device)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HIDDEN_DIM, HIDDEN_DIM, ATTN_DIM).to(device)

model = Seq2Seq(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

src = torch.tensor([[1,2,3,4,0],
                    [2,3,4,5,0]], dtype=torch.long).to(device)

trg = torch.tensor([[0,4,3,2,1],
                    [0,5,4,3,2]], dtype=torch.long).to(device)

EPOCHS = 200

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    output = model(src, trg)

    output = output[:, 1:].reshape(-1, OUTPUT_DIM)
    trg_y = trg[:, 1:].reshape(-1)

    loss = criterion(output, trg_y)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


model.eval()

with torch.no_grad():
    test_input = torch.tensor([[1,2,3,4,0]], dtype=torch.long).to(device)

    encoder_outputs, hidden = model.encoder(test_input)

    input_token = torch.tensor([0]).to(device)

    result = []

    for _ in range(4):
        output, hidden, attn = model.decoder(
            input_token, hidden, encoder_outputs
        )

        top1 = output.argmax(1)
        result.append(top1.item())

        input_token = top1

    print("Input: ", test_input.tolist())
    print("Predicted reversed:", result)