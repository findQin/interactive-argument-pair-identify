import os
import sys
os.chdir(sys.path[0])

from typing import Dict
import argparse
import json
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers.optimization import get_constant_schedule

from data import Data
from evaluate import evaluate, calculate_accuracy_f1, get_labels_from_file
from model import RnnForSentencePairClassification
from utils import get_csv_logger, get_path, get_device
from vocab import build_vocab


MODEL_MAP = {
    'rnn': RnnForSentencePairClassification
}


class Trainer:
    """Trainer for SMP-CAIL2020-Argmine.
    """

    def __init__(self,
                 model, data_loader: Dict[str, DataLoader], device, n_gpu, config):

        self.model = model
        self.device = device
        self.n_gpu = n_gpu
        self.config = config
        self.data_loader = data_loader

        self.num_training_steps = config.num_epoch * (
            len(data_loader['train']) )

        print("Num_training_steps: {}".format(self.num_training_steps))

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.criterion = nn.CrossEntropyLoss()

        
    def _get_optimizer(self):
        """Get optimizer for different models.

        Returns:
            optimizer
        """
        
        optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        
        return optimizer


    def _get_scheduler(self):
        """Get scheduler for different models.

        Returns:
            scheduler
        """
        scheduler = get_constant_schedule(self.optimizer)
        
        return scheduler


    def _evaluate_for_train_valid(self):
        """Evaluate model on train and valid set and get acc and f1 score.

        Returns:
            train_acc, train_f1, valid_acc, valid_f1
        """
        train_predictions = evaluate(
            model = self.model, data_loader = self.data_loader['valid_train'],
            device = self.device)

        valid_predictions = evaluate(
            model = self.model, data_loader = self.data_loader['valid_valid'],
            device = self.device)

        train_answers = get_labels_from_file(self.config.train_file_path)
        valid_answers = get_labels_from_file(self.config.valid_file_path)

        train_acc, train_f1 = calculate_accuracy_f1(
            train_answers, train_predictions)

        valid_acc, valid_f1 = calculate_accuracy_f1(
            valid_answers, valid_predictions)

        return train_acc, train_f1, valid_acc, valid_f1


    def _epoch_evaluate_update_description_log(
            self, tqdm_obj, logger, epoch):

        # Evaluate model for train and valid set
        results = self._evaluate_for_train_valid()
        train_acc, train_f1, valid_acc, valid_f1 = results

        # Update tqdm description for command line
        tqdm_obj.set_description(
            'Epoch: {:d}, train_acc: {:.6f}, train_f1: {:.6f}, '
            'valid_acc: {:.6f}, valid_f1: {:.6f}, '.format(
                epoch, train_acc, train_f1, valid_acc, valid_f1))

        # Logging
        logger.info(','.join([str(epoch)] + [str(s) for s in results]))

        return train_acc, train_f1, valid_acc, valid_f1


    def train(self):
        """Train model on train set and evaluate on train and valid set.

        Returns:
            state dict of the best model with highest valid f1 score
        """

        epoch_logger = get_csv_logger(
            os.path.join(self.config.log_path,
                         self.config.experiment_name + '-epoch.csv'),
            title='epoch,train_acc,train_f1,valid_acc,valid_f1')

        trange_obj = trange(self.config.num_epoch, desc='Epoch', ncols=100)

        self._epoch_evaluate_update_description_log(
            tqdm_obj=trange_obj, logger=epoch_logger, epoch=0)

        best_model_state_dict, best_valid_f1, global_step = None, 0, 0

        for epoch, _ in enumerate( trange_obj ):
            self.model.train()
            tqdm_obj = tqdm(self.data_loader['train'], ncols=80)

            for step, batch in enumerate( tqdm_obj ):

                batch = tuple( t.to( self.device ) for t in batch )

                sc_ids, bc_ids, sc_len, bc_len, label_ids = batch

                logits = self.model(s1_ids=sc_ids, 
                                    s2_ids=bc_ids,
                                    s1_lengths=sc_len,
                                    s2_lengths=bc_len
                                    )  

                loss = self.criterion(logits, label_ids)

                if self.n_gpu > 1:
                    loss = loss.mean()

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                global_step += 1

                tqdm_obj.set_description('loss: {:.6f}'.format(loss.item()))

            results = self._epoch_evaluate_update_description_log(
                tqdm_obj=trange_obj, logger=epoch_logger, epoch=epoch + 1)
            
            if results[-1] > best_valid_f1:
                best_model_state_dict = deepcopy(self.model.state_dict())
                best_valid_f1 = results[-1]

        return best_model_state_dict


def main(config_file='config/rnn_config.json'):

    # 0. Load config and mkdir
    with open(config_file) as fin:
        config = json.load(fin, object_hook=lambda d: SimpleNamespace(**d))

    get_path(config.log_path)
    
    build_vocab(file_in=config.all_train_file_path,
                    file_out=os.path.join(config.model_path, 'vocab.txt'))

    # 0.1. Apply device
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split(',')] 
    print("gpu_ids:{}".format(gpu_ids))
    
    device, n_gpu = get_device(gpu_ids[0])

    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    # 1. Load data
    data = Data(vocab_file=os.path.join(config.model_path, 'vocab.txt'),
                max_seq_len=config.max_seq_len,
                model_type=config.model_type)

    datasets = data.load_train_and_valid_files(
        train_file=config.train_file_path,
        valid_file=config.valid_file_path)

    train_set, valid_set_train, valid_set_valid = datasets

    sampler_train = RandomSampler(train_set)

    data_loader = {
        'train': DataLoader(
            train_set, sampler = sampler_train, batch_size = config.batch_size),
        'valid_train': DataLoader(
            valid_set_train, batch_size=config.batch_size, shuffle=False),
        'valid_valid': DataLoader(
            valid_set_valid, batch_size=config.batch_size, shuffle=False)}

    config.device = device

    # 2. Build model
    model = MODEL_MAP[config.model_type](config)
    model.to(device)

    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    # 3. Train
    trainer = Trainer(model=model, data_loader=data_loader,
                      device=device, n_gpu=n_gpu, config=config)

    best_model_state_dict = trainer.train()

    # 4. Save model
    torch.save(best_model_state_dict,
               os.path.join(config.model_path, 'model.bin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config_file', default='config/rnn_config.json',
        help='model config file')

    args = parser.parse_args()
    main(args.config_file)
