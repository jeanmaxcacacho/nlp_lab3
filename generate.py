import torch
import pickle
import torchtext
from create_model import Transformer, TransformerDecoder # do not remove unused import

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('model 30 2024-04-21 21-04-46,549638.pth').to(DEVICE)
model.eval()

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

sample = "[SOH]"
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
sample = tokenizer(sample)

initial = vocab.lookup_indices(sample)

softmax = torch.nn.Softmax(dim=0)

context_window = initial[::]

# 5 + 7 + 5 + 3 (comma, comma, period) + 1 ([soh]) = 21
while len(initial) < 21:
    data = torch.Tensor(context_window).unsqueeze(0).to(device=DEVICE, dtype=torch.int)
    pred = model(data)
    pred = softmax(pred.squeeze())
    pred = torch.multinomial(pred, 1)[0].item()
    if vocab.lookup_token(initial[-1]) == ".":
        print('End of haiku token found! Stopping generation prematurely')
        break

    context_window.append(pred)

    if len(context_window) >= 4:
        context_window = context_window[1:]

    initial.append(pred)

initial = vocab.lookup_tokens(initial)
for i,e in enumerate(initial):
    # if the next token are any of those characters don't output a space
    if i < len(initial) - 1 and initial[i+1] not in ['.',',']:
        initial[i] = e + ' '
    if e == ',':
        initial[i] = e + '\n'
print(''.join(initial))