import torch
import codecs

import collections
import pandas as pd
from tqdm import tqdm
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)


class InputExample(object):
    """A single training/test example for dataset."""
    '''
    For  dataset:
    start_ending: question
    ending_0/1/2/3/4: option_0/1/2/3/4
    label: true answer
    '''
    def __init__(self,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 ending_4,
                 label = None):

        self.start_ending = start_ending

        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
            ending_4
        ]

        self.label = label



class InputFeatures(object):
    def __init__(self,
                 choices_features,
                 label
    ):
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label



def read_examples(paths, train):
    examples = []
    data_frame = pd.read_csv(paths)

    for row in data_frame.itertuples(index=False):
        candidates = row[3:8]
        answer = int(row[-1])-1 if train else None
        sc = row[2]

        examples.append(
            InputExample(
                start_ending = sc,
                ending_0 = candidates[0],
                ending_1 = candidates[1],
                ending_2 = candidates[2],
                ending_3 = candidates[3],
                ending_4 = candidates[4],
                label=answer
            ))

    return examples 



def _truncate_seq(tokens_a, max_length):
    """Truncates question in place to the maximum question length."""

    while True:
        total_length = len(tokens_a) 
        if total_length <= max_length:
            break
        else:
            tokens_a.pop()



def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    max_query_length=259
    max_answer_length=250 

    features = []
    for _, example in enumerate( tqdm( examples, ncols=80, desc = "convert examples to features") ):
        start_ending_tokens = tokenizer.tokenize(example.start_ending)
        _truncate_seq(start_ending_tokens, max_query_length)

        choices_features = []
        for _, ending in enumerate(example.endings):
            ending_tokens = tokenizer.tokenize(ending)
            _truncate_seq(ending_tokens, max_answer_length)

            tokens = ["[CLS]"] + start_ending_tokens + ["[SEP]"] + ending_tokens + ["[SEP]"]
            
            segment_ids = [0] * (len(start_ending_tokens) + 2) + [1] * (len(ending_tokens) + 1)
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((input_ids, input_mask, segment_ids))
    
        label = example.label

        features.append(
            InputFeatures(
                choices_features = choices_features,
                label = label
            )
        )

    return features



def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]



def convert_features_to_tensors(features, batch_size, train):
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)

    if train:  
        all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask,
                        all_segment_ids, all_label_ids)

        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    else:
        dataset = TensorDataset(all_input_ids, all_input_mask,
                        all_segment_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader



def get_labels_from_file(filename):
    """Get labels on the last column from file.
    Args:
        filename: file name
    Returns:
        List[str]: label list
    """
    labels = []
    with codecs.open(filename, 'r', encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            label = int(line.strip().split(',')[-1])-1
            labels.append(label)

    return labels          




def load_data(path, tokenizer, max_seq_length, batch_size, train=False):

    examples = read_examples(path, train)
    features = convert_examples_to_features(examples, tokenizer, max_seq_length)
    dataloader = convert_features_to_tensors(features, batch_size, train)
    
    if train:
        return dataloader, len(dataloader)
    else:
        labels = get_labels_from_file(path)
        return dataloader, labels



if __name__ == "__main__":
    
    print(123456)