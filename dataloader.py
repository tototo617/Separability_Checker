import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import utils
import yaml


class transform():
    def __init__(self,config):
        self.wp = utils.wav_processor(config)

    def __call__(self,y,y2=None,is_train=True):
        logpow = (self.wp.log_power(y))
        logpow = torch.tensor(logpow,dtype=torch.float32)

        if is_train:
            rand = self.rand()

            if rand==1:
                logpow = torch.cat([logpow,torch.roll(logpow,np.random.randint(200,8000))],dim=1)
                target = torch.zeros([1],dtype=torch.int64)
            else:
                logpow = torch.cat([logpow,torch.roll(-logpow,np.random.randint(200,8000))],dim=1)
                target = torch.ones([1],dtype=torch.int64)

            return logpow, target
        
        else:
            logpow2 = (self.wp.log_power(y2))
            logpow2 = torch.tensor(logpow,dtype=torch.float32)

            logpow = torch.cat([logpow,torch.roll(-logpow,np.random.randint(200,8000))],dim=1)

            return logpow


    def rand(self):
        return np.random.randint(0,2)


class wav_dataset(Dataset):
    def __init__(self,config,path_scp):
        self.wp = utils.wav_processor(config)
        
        self.scp = self.wp.read_scp(path_scp)
        self.keys = [key for key in self.scp.keys()]
        self.trans = transform(config)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        y = self.wp.read_wav(self.scp[key])

        y_trans = self.trans(y)
        
        return y_trans


def padding(batch):
    batch_logpow,batch_target = [],[]
    for logpow,target in batch:
        batch_logpow.append(logpow)
        batch_target.append(target)

    batch_logpow = pad_sequence(batch_logpow, batch_first=True)
    batch_target = torch.tensor(batch_target)

    return batch_logpow, batch_target


def make_dataloader(config,path_scp):
    dataset  = wav_dataset(config, path_scp)

    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']
    shuffle = config['dataloader']['shuffle']

    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,
                                shuffle=shuffle,collate_fn=padding)

    return dataloader

if __name__ == "__main__":
    with open('./config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    dataloader = make_dataloader(config)

    for logpow, target in dataloader:
        print(logpow,target)