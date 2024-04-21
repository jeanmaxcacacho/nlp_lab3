import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import pickle
import pandas as pd
import datetime

class TransformerDecoder(nn.Module):
    '''
    Follows GPT-1 architecture, hence the use of decoder layers
    https://en.wikipedia.org/wiki/GPT-1#/media/File:Full_GPT_architecture.svg
    '''
    def __init__(self, d_model=24, n_head=4, d_ffn=128, dropout=0.1, device='cpu'):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_ffn = d_ffn
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.ffn_1 = nn.Linear(d_model, d_ffn)
        self.ffn_2 = nn.Linear(d_ffn, d_model)

        self.gelu = nn.GELU()

        self.attention = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True, device=device)
    
    def forward(self, x):
        # x is of shape (N,L,d_model)

        x_1 = self.norm_1(x) # remember x for residual

        # generate mask for masked self-attention
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(self.device)

        x_1, _ = self.attention(x_1, x_1, x_1, attn_mask=mask, need_weights=False)

        x_1 = self.dropout(x_1)
        x_1 = x_1 + x
        x_1 = self.norm_2(x_1)

        x_2 = self.ffn_1(x_1)
        x_2 = self.gelu(x_2)
        x_2 = self.ffn_2(x_2)
        x_2 = self.dropout(x_2)

        return x_2 + x_1

class Transformer(nn.Module):
    def __init__(self, context_size, vocab_size, d_model=24, dropout=0.1, n_block=4, device='cpu'):
        super(Transformer, self).__init__()
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ffn = 2048
        self.dropout = nn.Dropout(dropout)
        self.n_block = n_block
        self.device = device

        self.embedding = nn.Embedding(vocab_size, d_model)
    
        self.dec_blocks = nn.ModuleList([
            TransformerDecoder(d_model=d_model,dropout=dropout, device=device) for _ in range(n_block)
        ])

        self.pe = self.gen_pe(context_size, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Linear(d_model, vocab_size)
    
    def gen_pe(self, r, c):
        pe = torch.zeros(r, c).to(self.device)
        for k in range(r):
            for i in range(c):
                if i % 2 == 0:
                    # theta = k / (10_000 ** (i / c)) # overflow error, fixed using log-exp trick
                    theta = math.e ** ((-i/c) * math.log(10_000))
                    pe[k,i] = math.sin(k * theta)
                else:
                    # theta = k / (10_000 ** ((i-1) / c))
                    theta = math.e ** (((-i+1)/c) * math.log(10_000))
                    pe[k,i] = math.cos(k * theta)
        return pe

    def forward(self, x):
        # x is of shape (N,L,d_model)

        x = self.embedding(x)
        length = x.shape[1]
        x = x + self.pe[:length]
        x = self.dropout(x)

        for dec in self.dec_blocks:
            x = dec(x)
        
        x = self.ffn(x[:, -1])

        # no softmax as we use CELoss
        return x
    
class CustomDataset(Dataset):
    def __init__(self, x):
        self.x = torch.Tensor(x)
        self.n_samples = len(x)
    
    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.n_samples

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv('dataset.csv', index_col=0, dtype='long')

    context_size = len(df.columns)

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)

    def train_fn(loader, model, optimizer, loss_fn, device='cpu'):
        loop = tqdm(loader, position=0)
        ave_loss = 0
        count = 0
        for _, (data) in enumerate(loop):
            data = data.to(device=device, dtype=torch.long)

            context_size = data.shape[-1]

            ave_loss_1 = 0
            count_1 = 0

            for i in tqdm(range(2, context_size), position=1, desc="inner loop", leave=False):
                d_ = data[:, :i]
                tgt_i = d_[:, -1]
                d_ = d_[:, :-1]

                tgt = torch.zeros(d_.shape[0], model.vocab_size).to(device=device)
                rows = torch.arange(len(tgt_i)).to(device=device)
                tgt[rows, tgt_i.view(-1)] = 1

                pred = model(d_)

                loss = loss_fn(pred, tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_postfix(loss=loss.item())
                ave_loss_1 += loss.item()
                count_1 += 1
            
            # remove later
            if count_1 == 0:
                print('Division by zero found, check the loop bounds logic')
            else:
                ave_loss_1 = ave_loss_1 / count_1

            ave_loss += ave_loss_1
            count += 1
        
        ave_loss = ave_loss / count
        return ave_loss
    
    dataset = CustomDataset(df.values)
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = True
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

    read_from_file = 0
    if read_from_file:
        # replace the name
        model = torch.load('model 30 2024-04-21 20-54-40,519798.pth')
    else: 
        model = Transformer(context_size, vocab_size, device=DEVICE).to(device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    NUM_EPOCHS = 40

    curr_time = str(datetime.datetime.now()).replace(':','-').replace('.',',')

    last_stopped_epoch = 1

    for i in range(NUM_EPOCHS):
        ave_loss = train_fn(loader, model, optimizer, criterion, device=DEVICE)
        curr_epoch = last_stopped_epoch + i
        print(f'Epoch {curr_epoch}: {ave_loss}')
        if curr_epoch % 5 == 0:
            torch.save(model, 'model' + ' ' + str(curr_epoch) + ' ' + curr_time + '.pth')