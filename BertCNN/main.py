import json
import os
from types import SimpleNamespace

import fire
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import Data
from evaluate import evaluate
from model import BertForClassification
from utils import load_torch_model,  get_device


LABELS = ['1', '2', '3', '4', '5']
MODEL_MAP = {
    'bert': BertForClassification,
}


def main(in_file = '../work/lbwj_bin/data/test.csv',
         out_file = 'output/result1.csv',
         model_config = 'config/bert_config.json'):

    # 0. Load config
    with open(model_config) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split(',')] 

    device, n_gpu = get_device( gpu_ids[0] )

    # 1. Load data
    data = Data(vocab_file = os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len = config.max_seq_len,
                model_type = config.model_type)

    test_set = data.load_file(in_file, train=False)

    data_loader_test = DataLoader(
        test_set, batch_size = config.batch_size, shuffle = False)

    # 2. Load model
    model = MODEL_MAP[config.model_type](config)

    model = load_torch_model(
        model, model_path=os.path.join(config.model_path, 'model.bin'))

    model.to(device)

    # 3. Evaluate
    answer_list = evaluate(model, data_loader_test, device)

    # 4. Write answers to file
    id_list = pd.read_csv(in_file)['id'].tolist()
    with open(out_file, 'w') as fout:
        fout.write('id,answer\n')
        for i, j in zip(id_list, answer_list):
            fout.write(str(i) + ',' + str(j) + '\n')


if __name__ == '__main__':
    fire.Fire(main)
