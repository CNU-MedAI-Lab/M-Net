from xlstm2 import sLSTM
from itertools import pairwise
import torch

seq_len = 15
batch_size = 64

inp_dim = 10*10
head_dim = 8
head_num = 4

# Create a mock up input sequence
seq = torch.randn(seq_len, batch_size, inp_dim)

lstm = sLSTM(
    inp_dim,        # Input sequence dimension
    head_dim,       # Dimension of each head
    head_num,       # Number of heads
    p_factor=4/3,   # Tunable expansion factor
)

# Initialize the hidden states
hid = lstm.init_hidden(batch_size)

criterion = ... # Pick some loss function, i.e. MSE

# Iterate through the sequence length
loss = 0
outs = []
print(seq.shape)
i = 0
for prev in (seq):
    print(i)
    i+=1
    # Get the model prediction plus the updated hidden states
    # print(prev.shape)
    pred, hid = lstm(prev, hid)
    pred = pred.view(1, pred.shape[0], pred.shape[1])
    # print(prev.shape)
    outs.append(pred)
print(len(outs))
outs = torch.cat(outs, 0)
print(outs.shape)
