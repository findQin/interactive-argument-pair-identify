"""Evaluate model and calculate results for SMP-CAIL2020-Argmine.

Author: Yixu GAO yxgao19@fudan.edu.cn
"""
from typing import List
import codecs

from sklearn import metrics


LABELS = ['1', '2', '3', '4', '5']


def calculate_accuracy_f1(
        golds: List[str], predicts: List[str]) -> tuple:
    """Calculate accuracy and f1 score.

    Args:
        golds: answers
        predicts: predictions given by model

    Returns:
        accuracy, f1 score
    """
    return metrics.accuracy_score(golds, predicts), \
           metrics.f1_score(
               golds, predicts,
               labels=LABELS, average='macro')


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
            labels.append(line.strip().split(',')[-1])
    return labels


def eval_file(golds_file, predicts_file):
    """Evaluate submission file

    Args:
        golds_file: file path
        predicts_file:  file path

    Returns:
        accuracy, f1 score
    """
    golds = get_labels_from_file(golds_file)
    predicts = get_labels_from_file(predicts_file)
    return calculate_accuracy_f1(golds, predicts)


if __name__ == '__main__':

    acc, f1_score = eval_file(
        'test.csv', 'output/test_result.csv')

    print("acc: {}, f1: {}".format(acc, f1_score))
