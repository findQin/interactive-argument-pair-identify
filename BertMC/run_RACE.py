import os
import sys
os.chdir(sys.path[0])
import argparse
import random
from tqdm import tqdm, trange
from copy import deepcopy
import collections
from collections import OrderedDict

import torch
import torch.nn as nn

from transformers import BertTokenizer
from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup )

from BertMC import BertOrigin
from data import load_data
from args import get_args
from sklearn import metrics


def train(epoch_num, 
        n_gpu, 
        train_dataloader,
        valid, 
        valid_train,
        model, 
        optimizer, 
        criterion, 
        max_grad_norm, 
        device, 
        scheduler, 
        output_model_path):

    best_model_state_dict, best_dev_f1, global_step = None, 0, 0  
    
    for epoch in range( int( epoch_num ) ):
        
        model.train()
        
        print(f'-------------- Epoch: {epoch+1:02} ----------')

        for step, batch in enumerate( tqdm( train_dataloader, desc = "Iteration" ) ):
        
            batch = tuple(t.to(device) for t in batch)
            
            # label_ids：标签
            input_ids, input_mask, segment_ids, label_ids = batch
            
            # logits : (batch_size, 5)
            # [[0.3, 0.2, 0.1, 0.2, 0.3], [0.4, 0.6,...]]
            logits = model(input_ids, segment_ids, input_mask)
            
            # label_id : (batch_size) 
            # [1, 1, ...]      
            loss = criterion(logits, label_ids)
                            
            if n_gpu > 1:
                loss = loss.mean()
            
            optimizer.zero_grad()
            # 反向传播
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
            
            # 更新参数                
            optimizer.step()
            # 更新学习率
            scheduler.step()               
            global_step += 1
       
        print('---traing is Ok----')
        valid_train_dataloader, valid_train_labels = valid_train

        valid_train_preds = evaluate(
            model, valid_train_dataloader,  device)

        train_acc, train_f1 = final_evaluate(valid_train_labels, valid_train_preds)
        
        dev_dataloader, dev_labels = valid

        dev_preds = evaluate(
            model, dev_dataloader,  device)

        dev_acc, dev_f1 = final_evaluate(dev_labels, dev_preds)
        
        s = 'Epoch: {:d}, train_acc: {:.6f}, train_f1: {:.6f}, ' \
            'valid_acc: {:.6f}, valid_f1: {:.6f}, '.format(
            epoch, train_acc, train_f1, dev_acc, dev_f1)
        
        print(s)

        with open(os.path.join(output_model_path,'save_run_result.txt'), 'a', encoding = 'utf-8') as f:
            f.write(s)
            f.write('\n')
                
        if dev_f1 > best_dev_f1: 
            best_model_state_dict = deepcopy(model.state_dict())
            best_dev_f1 = dev_f1
            
    return best_model_state_dict                                    



def evaluate(model, dataloader, device):
    # model.eval()中的数据不会进行反向传播，但是仍然需要计算梯度
    model.eval()
    answer_list = []

    for batch in tqdm(dataloader, desc = "Eval"):

        batch = tuple(t.to(device) for t in batch)
        
        input_ids, input_mask, segment_ids = batch
        
        # 数据不需要计算梯度，也不会进行反向传播
        # 进一步加速和节省gpu空间
        with torch.no_grad():        
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            # logits: (batch_size, num_choices)
        
        # (batch_size) tensor->list
        pred_ans = torch.argmax(logits, dim=-1).tolist()
        
        answer_list.extend(pred_ans)

    return answer_list



def final_evaluate(golds, preds):

    LABELS = [0, 1, 2, 3, 4]

    return metrics.accuracy_score(golds, preds), \
           metrics.f1_score(
               golds, preds,
               labels=LABELS, average='macro')



def get_device(gpu_id):

    device = torch.device("cuda:" + str(gpu_id)
                          if torch.cuda.is_available() else "cpu")                  
    n_gpu = torch.cuda.device_count()
    
    if torch.cuda.is_available():
        print("device is cuda, # cuda amount is: ", n_gpu)  
    else:
        print("device is cpu, not recommend")
        
    return device, n_gpu



def main(config, bert_vocab_file, do_prediction=False):
        
    # --gpu_ids: [1,2,3]--
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split(',')] 
    print("gpu_ids:{}".format(gpu_ids))
    
    device, n_gpu = get_device(gpu_ids[0])

    if n_gpu > 1:
        n_gpu = len(gpu_ids)
    
    print("n_gpu:{}".format(n_gpu))

    tokenizer = BertTokenizer( vocab_file =  bert_vocab_file )       
    label_list = ["0", "1", "2", "3", "4"]

    criterion = nn.CrossEntropyLoss()   
    criterion = criterion.to(device)

    if not do_prediction:
        # 数据准备
        train_file = os.path.join(config.data_dir, "train.csv")   
        dev_file = os.path.join(config.data_dir, "valid.csv")

        train_dataloader, train_len = load_data(train_file, tokenizer, config.max_seq_length, config.batch_size, train=True)
        print("Num train_set: {}".format(train_len))
        
        valid_train = load_data(train_file, tokenizer, config.max_seq_length, config.batch_size)
        print("Num valid_train_set: {}".format(len(valid_train[0])))

        valid = load_data(dev_file, tokenizer, config.max_seq_length, config.batch_size)
        print("Num dev_set: {}".format(len(valid[0])))
        
        num_train_steps = int(
            train_len  * config.num_train_epochs)

        num_warmup = int(num_train_steps * config.warmup_ratio)

        print("num_train_steps: {}".format(num_train_steps))
        print("num_warmup_steps: {}".format(num_warmup))
        
        # 模型准备
        model = BertOrigin(config, num_classes = 5)  
        model.to(device)   
        if n_gpu > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            
        no_decay = ['bias', 'gamma', 'beta']
            
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}]
                
        optimizer = AdamW(
            optimizer_parameters,
            lr = config.learning_rate,
            betas = (0.9, 0.999),
            weight_decay = 1e-8,
            correct_bias = False)
                    
        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = num_warmup,
                num_training_steps = num_train_steps)

        best_model_state_dict = train(config.num_train_epochs, n_gpu, train_dataloader, valid, valid_train, model, optimizer, criterion,
                                   config.max_grad_norm, device, scheduler, config.output_dir)

        torch.save(best_model_state_dict, config.best_model_file)
    
    else:
        print('---**Enter Test**---')
        
        #dev_dataloader, dev_examples, dev_features, dev_labels = dev[:-1]

        test_file = os.path.join(config.data_dir, "test.csv")   
        test_dataloader, test_labels = load_data(
            test_file, tokenizer, config.max_seq_length, config.batch_size)
        
        print('Num test_dataloader: {}'.format(len(test_dataloader)))
        
        test_model = BertOrigin(config, num_classes = 5)
        
        pretrained_model_dict = torch.load(config.best_model_file)
        new_state_dict = OrderedDict()
        for k, value in pretrained_model_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = value        

        test_model.load_state_dict(new_state_dict, strict=True)
        test_model.to(device)

        if n_gpu > 1:
            test_model = nn.DataParallel(test_model, device_ids=gpu_ids)

        test_preds = evaluate(
            test_model, test_dataloader, device)

        test_acc, test_f1 = final_evaluate(test_labels, test_preds)
        
        print(f'\t  Acc: {test_acc*100: .3f}% | f1: {test_f1*100: .3f}%')
    


if __name__ == "__main__":

    data_dir = "../work/lbwj_bin/data/"   
    gpu_ids = "1,2"
    
    bert_vocab_file = "../work/lbwj_bin/model/bert/vocab.txt"
    
    # do_prediction: False(训练), True(预测)
    main(get_args(data_dir, gpu_ids), bert_vocab_file, do_prediction=True)
