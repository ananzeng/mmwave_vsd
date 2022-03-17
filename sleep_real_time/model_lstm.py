import torch
from tqdm import tqdm
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn

#input sequence length of 9 and unit number of 512
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length, dropout):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout

        self.lstm = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, dropout = dropout)

        self.fc = nn.Linear(hidden_size , num_classes)


    def forward(self, x):
        # Set initial hidden and cell states

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        """
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        """
        out, _ = self.lstm(
            x, h0
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #print("經過lstm", out.shape)
        out = out.reshape(out.shape[0], -1)
        #print("經過reshape", out.shape)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        #print("經過fc", out.shape)
        return out