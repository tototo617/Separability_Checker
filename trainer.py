import model
import utils
import yaml
import torch
import os
import dataloader
from tqdm import tqdm
import logging
import tensorboardX as tbx
import datetime

class Trainer():
    def __init__(self,model,SC_model):
        self.SC_model = SC_model
        self.cur_epoch = 0
        self.total_epoch = config['train']['epoch']
        self.early_stop = config['train']['early_stop']
        self.checkpoint = config['train']['path']
        self.name = config['name']

        opt_name = config['optim']['name']
        weight_decay = config['optim']['weight_decay']
        lr = config['optim']['lr']
        momentum = config['optim']['momentum']

        optimizer = getattr(torch.optim, opt_name)
        if opt_name == 'Adam':
            self.optimizer = optimizer(self.SC_model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer(self.SC_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.clip_norm = config['optim']['clip_norm'] if config['optim']['clip_norm'] else 0

        self.device = torch.device(config['gpu'])
        if config['train']['resume']['state']:    
            self.model_load(config)

        self.SC_model = SC_model.to(self.device)


    def train(self,dataloader,is_train):
        if is_train:
            self.SC_model.train()
        else:
            self.SC_model.eval()
        num_batchs = len(dataloader)
        total_loss = 0.0


        for logpow,target in tqdm(dataloader):
            logpow = logpow.to(self.device)
            target = target.to(self.device)

            output = self.SC_model(logpow)
            epoch_loss = (self.SC_model.loss(output,target)).to(self.device)
            total_loss += epoch_loss.item()

            if is_train:
                self.optimizer.zero_grad()
                epoch_loss.backward()
                if self.clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.SC_model.parameters(),self.clip_norm)
                self.optimizer.step()

        total_loss = total_loss/num_batchs
        
        return total_loss


    def run(self,train_dataloader,valid_dataloader):
        train_loss = []
        val_loss = []

        dt_now = datetime.datetime.now()
        writer = tbx.SummaryWriter("tbx/" + dt_now.isoformat())
        os.makedirs('./checkpoint/Separability_Checker',exist_ok=True)
        logging.basicConfig(filename='./checkpoint/Separability_Checker/train_log.log', level=logging.DEBUG)

        self.save_checkpoint(self.cur_epoch,best=False)
        v_loss = self.train(valid_dataloader,is_train=False)
        best_loss = v_loss
        no_improve = 0


        with torch.cuda.device(self.device):
            while self.cur_epoch < self.total_epoch:
                self.cur_epoch += 1
                t_loss = self.train(train_dataloader,is_train=True)
                print('epoch{0}:train_loss {1}'.format(self.cur_epoch,t_loss))
                logging.info('epoch{0}:train_loss {1}'.format(self.cur_epoch,t_loss))

                v_loss = self.train(valid_dataloader,is_train=False)
                print('epoch{0}:valid_loss {1}'.format(self.cur_epoch,v_loss))
                logging.info('epoch{0}:valid_loss {1}'.format(self.cur_epoch,v_loss))

                writer.add_scalar('t_loss', t_loss, self.cur_epoch)
                writer.add_scalar('v_loss', v_loss, self.cur_epoch)

                train_loss.append(t_loss)
                val_loss.append(v_loss)

                if v_loss >= best_loss:
                    no_improve += 1
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(self.cur_epoch,best=True)
                
                if no_improve == self.early_stop:
                    break
                self.save_checkpoint(self.cur_epoch,best=False)

        writer.close()

    def save_checkpoint(self, epoch, best=True):
        self.SC_model.to('cpu')
        print('save model epoch:{0} as {1}'.format(epoch,"best" if best else "last"))
        os.makedirs(os.path.join(self.checkpoint,self.name),exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.SC_model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        },
        os.path.join(self.checkpoint,self.name,'{0}.pt'.format('best' if best else 'last')))

        self.SC_model.to(self.device)

    def model_load(self,config):
        print('load on:',self.device)

        ckp = torch.load(config['train']['resume']['path'],map_location=torch.device('cpu'))
        self.cur_epoch = ckp['epoch']
        self.SC_model.load_state_dict(ckp['model_state_dict'])
        self.optimizer.load_state_dict(ckp['optim_state_dict'])

        self.SC_model = self.SC_model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        print('training resume epoch:',self.cur_epoch)


if __name__ == "__main__":
    with open('./config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    SC = model.Separability_Cheker(config)

    # wp = utils.wav_processor(config)
    # y = wp.read_wav("./00000.wav")
    # logpow = wp.log_power(y)
    # logpow = torch.tensor(logpow.T, dtype=torch.float32)
    # batch_logpow = torch.stack([logpow,logpow,logpow])

    path_train = "./scp/tr_s1.scp"
    path_valid = "./scp/cv_s1.scp"

    train_dataloader = dataloader.make_dataloader(config,path_train)
    valid_dataloader = dataloader.make_dataloader(config,path_valid)
    trainer = Trainer(config,SC)

    trainer.run(train_dataloader,valid_dataloader)



    