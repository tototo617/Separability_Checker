import torch
import dataloader
import model
import utils
import yaml
import logging
import numpy as np
from tqdm import tqdm
import datetime
import os


class Tester():
    def __init__(self,config,path_model):
        self.wp = utils.wav_processor(config)
        self.SC_model = model.Separability_Cheker(config)
        self.device = torch.device(config['gpu'])
        print('Processing on',config['gpu'])

        self.path_model = path_model
        ckp = torch.load(self.path_model,map_location=self.device)
        self.SC_model.load_state_dict(ckp['model_state_dict'])
        self.SC_model.eval()

        dt_now = datetime.datetime.now()
        self.path_test = config['test']['path_test'] + '/'+str(dt_now.isoformat())

        self.trans = dataloader.transform(config)
        


    def run(self):
        # os.makedirs(self.path_test,exist_ok=True)
        # logging.basicConfig(filename=self.path_test+'/logger.log', level=logging.DEBUG)
        # logging.info(self.path_model)

        path_test1 = "./scp/cv_s1.scp"
        path_test2 = "./scp/cv_s1.scp"

        scp_test1 = self.wp.read_scp(path_test1)
        scp_test2 = self.wp.read_scp(path_test2)
        softmax = torch.nn.Softmax()

        with torch.no_grad():
            for key in scp_test1.keys():
                y1 = self.wp.read_wav(scp_test1[key])
                y2 = self.wp.read_wav(scp_test2[key])

                testdata = self.trans(y1,y2,is_train=False)
                check_result = self.SC_model(testdata)
                print(key,np.round(np.array(softmax(check_result)), 5))



if __name__ == "__main__":
    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    tester = Tester(config,path)
    tester.run()