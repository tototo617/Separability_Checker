import torch
import torch.nn as nn
import utils
import yaml


class Separability_Cheker(nn.Module):
    def __init__(self,config):
        super().__init__()

        n_fft = config['transform']['n_fft']
        input_size = int(n_fft/2 + 1)*2
        hidden_size = config['network']['hidden_size']
        num_layers = config['network']['num_layer']
        dropout = config['network']['dropout']
        bidirectional = config['network']['bidirectional']

        self.num_unit = 2*hidden_size*num_layers if bidirectional else hidden_size*num_layers
        
        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout if num_layers > 1 else 0,
                            bidirectional = bidirectional)
                            
        self.dropout =  nn.Dropout(dropout)
        self.liner = nn.Linear(2*self.num_unit, 2)
        self.softmax = nn.LogSoftmax(dim=1)

        self.crossE_loss = nn.CrossEntropyLoss()


    def forward(self, x):
        if len(x.shape)==2:
            x = torch.unsqueeze(x, 0)

        B,T,F = x.shape

        output,(h_n,c_n) = self.lstm(x)

        params_lstm = torch.cat([h_n,c_n],dim=2)
        x = params_lstm.reshape([B,2*self.num_unit])

        x = self.dropout(x)
        x = self.liner(x)
        # x = self.softmax(x)

        return x

    def loss(self,output,target):

        # print(torch.sum(torch.abs(torch.argmax(output,axis=1)-target)))
        return self.crossE_loss(output,target)


if __name__ == "__main__":
    with open('./config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    SC = Separability_Cheker(config)

    wp = utils.wav_processor(config)
    y,sr = wp.read_wav("./00000.wav")
    logpow = wp.log_power(y)
    logpow = torch.tensor(logpow, dtype=torch.float32)
    batch_logpow = torch.stack([logpow,logpow,logpow])

    SC.eval()
    

    output = SC(batch_logpow)
    target = torch.tensor([0,1,0], dtype=torch.int64)

    loss = SC.loss(output,target)

    print(loss)