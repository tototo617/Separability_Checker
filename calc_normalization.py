import numpy as np
import yaml
from tqdm import tqdm
import pickle
import utils
import os

if __name__=="__main__":
    print('calc normalizing parameters')

    with open('config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    wp = utils.wav_processor(config)
    path_scp_train = "./scp/tr_s1.scp"
    scp_mix = wp.read_scp(path_scp_train)

    f_bin = int(config['transform']['n_fft']/2+1)

    mean_f = np.zeros(f_bin)
    var_f = np.zeros(f_bin)

    for key in tqdm(scp_mix.keys()):
        y = wp.read_wav(scp_mix[key])
        logpow = wp.log_power(y,normalize=False)

        mean_f += np.mean(logpow, 0)
        var_f += np.mean(logpow**2, 0)

    mean_f = mean_f / len(scp_mix.keys())
    var_f = var_f / len(scp_mix.keys())

    std_f = np.sqrt(var_f - mean_f**2)

    path_model_save = "./checkpoint/Separability_Checker"
    os.makedirs(path_model_save,exist_ok=True)
    path_normalize = path_model_save+'/dict_normalize.ark'

    with open(path_normalize, "wb") as f:
        normalize_dict = {"mean": mean_f, "std": std_f}
        pickle.dump(normalize_dict, f)
    print("Global mean: {}".format(mean_f))
    print("Global std: {}".format(std_f))