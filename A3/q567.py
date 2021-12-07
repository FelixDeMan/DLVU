# %%
from collections import defaultdict
from multiprocessing.context import assert_spawning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from data import load_ndfa

np.random.seed(10)

MAX_CHARS_PER_BATCH = 10000

def prepare_batches(x_train, i2w, w2i):
    dict_size = len(i2w)
    
    # w2i['.pad'] = 0
    pad_val = w2i['.pad']
    # w2i['.start'] = 1
    start_val = w2i['.start']
    # w2i['.end'] = 2
    end_val = w2i['.end']

    for x in x_train:
        x.insert(0, start_val)
        x.append(end_val)

    sizes = defaultdict(list)
    for x in x_train:
        sizes[len(x)].append(x)

    t_sizes = dict()
    for k, v in sizes.items():
        t_sizes[k] = torch.tensor(v, dtype=torch.long)
    
    batches = []
    for _, x_tensor in t_sizes.items():
        x_tensor_len, n_chars = x_tensor.shape
        start_pad = start_val * torch.ones(x_tensor_len, dtype=torch.long)
        
        shifted_input = x_tensor[:, 2:-1]
        zeros = torch.zeros(x_tensor_len, dtype=torch.long)
        end_pad = end_val * torch.ones(x_tensor_len, dtype=torch.long)
        y_tensor = torch.column_stack(
            [start_pad.T,shifted_input, zeros.T, end_pad.T]         
        )

        assert x_tensor.shape == y_tensor.shape

        batch_size = MAX_CHARS_PER_BATCH // n_chars
        x_batches = torch.split(x_tensor, batch_size)
        y_batches = torch.split(y_tensor, batch_size)
        
        # TODO probably there is a smarter way to do one hots over the whole dict
        y_oh_batches = list()
        for y in y_batches:
            b, chrs = y.shape
            one_hots = torch.zeros(b,chrs, dict_size, dtype=torch.long)
            for bi in range(y.shape[0]):
                y_one_hot = torch.zeros(chrs, dict_size, dtype=torch.long)
                for el in range(chrs):
                    y_one_hot[el][y[bi,el]] = 1
                one_hots[bi, :, :] = y_one_hot
            y_oh_batches.append(one_hots)
        assert len(x_batches) == len(y_oh_batches)
        batches.extend(list(zip(x_batches, y_oh_batches)))
    np.random.shuffle(batches)
    return batches

class recurNet(nn.Module):

    def __init__(self, embedding_dim = 32, hidden_size = 16, vocab_size = 15, num_layers=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.layer1 = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)

        self.layer2 = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_size, num_layers=num_layers, batch_first=True)
        
        self.layer3 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        # b, n_chrs = input.shape
        emb = self.layer1(input)
        
        # assert emb.shape == (b, n_chrs, self.embedding_dim)
                    
        lstm, (hn, cn) = self.layer2(emb)

        # assert lstm.shape == (b, n_chrs, self.hidden_size)
        output = self.layer3(lstm)
        # assert output.shape == (b, n_chrs, self.vocab_size)

        return output

def train(epochs, batches, device):
    net = recurNet()
    print(net)
    net.to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr = 0.01)
    
    running_loss = 0
    losses =[]
    data = list()
    for epoch in range(epochs):
        ts = time.perf_counter()
        for i, (x,one_hots) in enumerate(batches):
            x, one_hots = x.to(device), one_hots.to(device)
            
            optimizer.zero_grad()

            out = net(x).softmax(dim=1)
            
            # print(f'{out.shape}=')
            # print(f'{one_hots.shape}=')

            loss = criterion(out, one_hots.type(torch.float32))
            
            # divide by batch and # of tokens
            loss /= one_hots.shape[0]* one_hots.shape[1]
            
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            losses.append(loss.item()) 
            if i % 50 == 0: #print every 1000 batches
                print('[%d, %5d] loss: %.3f ' %
                    (epoch +1, i+1, running_loss / 50))
                running_loss = 0.0
            data.append({'update' : i, 'epoch': epoch, 'loss': loss.item()})
        print(f'Epoch took: {time.perf_counter()-ts}s')
    print('Finished training')
    return losses, data,net


# %%
x_train, (i2w, w2i) = load_ndfa(n=150_000)
print(''.join([i2w[i] for i in    x_train[149_000]]) )
print(i2w)

batches = prepare_batches(x_train, i2w, w2i)



if torch.cuda.is_available():
    device = torch.device('cuda') 
else:
    device = torch.device('cpu') 

print(f'Training on: {device}', flush=True)

# %%
losses, data, net = train(10, batches, device)

import pandas as pd
df = pd.DataFrame(data)
df.groupby(by='epoch').mean()
# %%
df.groupby(by='epoch').mean()['loss'].plot()
# %%


    
def evaluate(net, batches):
    import random
    i = random.choice(list(range(len(batches))))
    batch, true_outs = batches[i]
    with torch.no_grad():
        out = net(batch).softmax(dim=1).argmax(1)
        true_labs = true_outs.argmax(1)
        res = []
        for i in range(batch.shape[0]):
            tot = len(out[i])
            hit = 0
            for p, t in zip(out[i], true_labs[i]):
                if p == t:
                    hit +=1 
            score = float(hit)/tot
            print(score)
            res.append(score)

    return res

        


# %%
evaluate(net, batches)
# %%
