"""Data processor for SMP-CAIL2020-Argmine.

In data file, each line contains 1 sc sentence and 5 bc sentences.
The data processor convert each line into 5 samples,
each sample with 1 sc sentence and 1 bc sentence.

Usage:
1. Tokenizer (used for RNN model):
    from data import Tokenizer
    vocab_file = 'vocab.txt'
    sentence = '我饿了，想吃东西了。'

    tokenizer = Tokenizer(vocab_file)
    tokens = tokenizer.tokenize(sentence)
    # ['我', '饿', '了', '，', '想', '吃', '东西', '了', '。']
    ids = tokenizer.convert_tokens_to_ids(tokens)
"""

from typing import List
import jieba
import torch
import re
import pandas as pd

from torch.utils.data import TensorDataset
from tqdm import tqdm,trange



class Tokenizer:
    """Tokenizer for Chinese given vocab.txt.
    Attributes:
        dictionary: Dict[str, int], {<word>: <index>}
    """

    def __init__(self, vocab_file='vocab.txt'):
        """Initialize and build dictionary.
        Args:
            vocab_file: one word each line
        """

        self.dictionary = {'[PAD]': 0, '[UNK]': 1}
        count = 2

        with open(vocab_file, encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                self.dictionary[word] = count
                count += 1


    def __len__(self):
        return len(self.dictionary)


    @staticmethod
    def tokenize(sentence: str) -> List[str]:
        """Cut words for a sentence.
        Args:
            sentence: sentence
        Returns:
            words list
        """
        return jieba.lcut(sentence)


    def convert_tokens_to_ids(
            self, tokens_list: List[str]) -> List[int]:
        """Convert tokens to ids.
        Args:
            tokens_list: word list
        Returns:
            index list
        """

        return [self.dictionary.get(w, 1) for w in tokens_list]




class Data:
    """Data processor for RNN model .
    Attributes:
        model_type: 'rnn'
        max_seq_len: int, default: 512
        tokenizer: Tokenizer for rnn
    """

    def __init__(self,
                 vocab_file='',
                 max_seq_len: int = 512,
                 model_type: str = 'rnn'):
        """Initialize data processor for SMP-CAIL2020-Argmine.
        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
            model_type: 'rnn' use Tokenizer as tokenizer
        """

        self.model_type = model_type
        self.tokenizer = Tokenizer(vocab_file)
        self.max_seq_len = max_seq_len


    def data_preprocess(self, arg):

        rule0 = r"^，"
        myrule0 = re.search(rule0, arg)
        if myrule0:
            arg = re.sub(rule0, "", arg)

        rule1 = r"时间时间时间，?"
        myrule1 = re.search(rule1, arg)
        if myrule1:
            arg = re.sub(rule1, "", arg)

        rule2 = r"\d+\s*[.、，,]"
        myrule2 = re.search(rule2, arg)
        if myrule2:
            arg = re.sub(rule2, "", arg)

        #rule2_1 = r"\d+[元|天]"
        #myrule2_1 = re.search(rule2_1, arg)
        #if myrule2_1:
        #    arg = re.sub(rule2_1, "", arg)

        #rule3 = r'（.*?）'
        #myrule3 = re.search(rule3, arg)
        #if myrule3:
        #    arg = re.sub(rule3, "", arg)

        #rule4 = r'自诉代[^理]'
        #myrule4 = re.search(rule4, arg)
        #if myrule4:
        #    arg = re.sub(r'自诉代', "自诉代理人", arg)

        #rule5 = r'第?[^人][一二三四五六七八九十][，、]'
        #myrule5 = re.search(rule5, arg)
        #if myrule5:
        #    arg = re.sub(rule5, "", arg)


        
        return arg


    def load_file(self,
                  file_path='SMP-CAIL2020-train.csv',
                  train=True) -> TensorDataset:
        """Load SMP-CAIL2020-Argmine train file and construct TensorDataset.
        Args:
            file_path: train file with last column as label
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label
        Returns:

            RNN model:
            Train:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
                torch.utils.data.TensorDataset
                    each record: (s1_ids, s2_ids, s1_length, s2_length)
        """
        sc_list, bc_list, label_list = self._load_file(file_path, train)

        dataset = self._convert_sentence_pair_to_rnn_dataset(
            sc_list, bc_list, label_list)

        return dataset


    def load_train_and_valid_files(self, train_file, valid_file):
        """Load all files for SMP-CAIL2020-Argmine.
        Args:
            train_file, valid_file: files for SMP-CAIL2020-Argmine

        Returns:
            train_set, valid_set_train, valid_set_valid
            all are torch.utils.data.TensorDataset
        """

        print('Loading train records for train...')
        train_set = self.load_file(train_file, True)
        print(len(train_set), 'training records loaded.')

        print('Loading train records for valid...')
        valid_set_train = self.load_file(train_file, False)
        print(len(valid_set_train), 'train records loaded.')

        print('Loading valid records...')
        valid_set_valid = self.load_file(valid_file, False)
        print(len(valid_set_valid), 'valid records loaded.')
        
        return train_set, valid_set_train, valid_set_valid
        

    def _load_file(self, filename, train: bool = True):
        """Load SMP-CAIL2020-Argmine train/test file.
        For train file,
        The ratio between positive samples and negative samples is 1:4
        Copy positive 3 times so that positive:negative = 1:1
        Args:
            filename: SMP-CAIL2020-Argmine file
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label
        Returns:
            sc_list, bc_list, label_list with the same length
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: List[int], list of labels
        """

        data_frame = pd.read_csv(filename)
        sc_list, bc_list, label_list = [], [], []

        for row in data_frame.itertuples(index=False):
            candidates = row[3:8]
            answer = int(row[-1]) if train else None
            sc = self.data_preprocess( row[2] )
            sc_tokens = self.tokenizer.tokenize(sc)

            for i, _ in enumerate(candidates):
                bc = self.data_preprocess( candidates[i] )
                bc_tokens = self.tokenizer.tokenize(bc)

                if train:
                    if i + 1 == answer:
                        # Copy positive sample 4 times
                        for _ in range(len(candidates) - 1):
                            sc_list.append(sc_tokens)
                            bc_list.append(bc_tokens)
                            label_list.append(1)
                    else:
                        sc_list.append(sc_tokens)
                        bc_list.append(bc_tokens)
                        label_list.append(0)

                else:  # test
                    sc_list.append(sc_tokens)
                    bc_list.append(bc_tokens)

        return sc_list, bc_list, label_list


    def _convert_sentence_pair_to_rnn_dataset(
            self, s1_list, s2_list, label_list=None):
        """Convert sentences pairs to dataset for RNN model.
        Args:
            sc_list, bc_list: List[List[str]], list of word tokens list
            label_list: train: List[int], list of labels
                        test: []
        Returns:
            Train:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
            Test:
            torch.utils.data.TensorDataset
                each record: (s1_ids, s2_ids, s1_length, s2_length, label)
        """
        all_s1_ids, all_s2_ids = [], []
        all_s1_lengths, all_s2_lengths = [], []

        for i in trange(len(s1_list), ncols=80):           
            tokens_s1, tokens_s2 = s1_list[i], s2_list[i]

            all_s1_lengths.append(min(len(tokens_s1), self.max_seq_len))
            all_s2_lengths.append(min(len(tokens_s2), self.max_seq_len))

            if len(tokens_s1) > self.max_seq_len:
                tokens_s1 = tokens_s1[:self.max_seq_len]
            if len(tokens_s2) > self.max_seq_len:
                tokens_s2 = tokens_s2[:self.max_seq_len]

            s1_ids = self.tokenizer.convert_tokens_to_ids(tokens_s1)
            s2_ids = self.tokenizer.convert_tokens_to_ids(tokens_s2)

            if len(s1_ids) < self.max_seq_len:
                s1_ids += [0] * (self.max_seq_len - len(s1_ids))
            if len(s2_ids) < self.max_seq_len:
                s2_ids += [0] * (self.max_seq_len - len(s2_ids))

            all_s1_ids.append(s1_ids)
            all_s2_ids.append(s2_ids)

        all_s1_ids = torch.tensor(all_s1_ids, dtype=torch.long)
        all_s2_ids = torch.tensor(all_s2_ids, dtype=torch.long)

        all_s1_lengths = torch.tensor(all_s1_lengths, dtype=torch.long)
        all_s2_lengths = torch.tensor(all_s2_lengths, dtype=torch.long)

        if label_list:  # train
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            
            return TensorDataset(
                all_s1_ids, all_s2_ids, all_label_ids)
        # test
        return TensorDataset(
            all_s1_ids, all_s2_ids)


