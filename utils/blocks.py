import pandas as pd
import numpy as np
import random
from sklearn.model_selection import ShuffleSplit



def build_train_test(base_data):
    rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=2021)
    for train_index, test_index in rs.split(base_data):
        yield train_index, test_index


def build_metrics(y_pred, y_test, task_type='binary-clf'):
    if task_type == 'binary-clf':
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            roc_auc_score,
            precision_score,
            roc_curve
        )
        acc_score = round(accuracy_score(y_test, y_pred),4)
        F1_score = round(f1_score(y_test, y_pred), 4)
        auc_score = round(roc_auc_score(y_test, y_pred), 4)
        prec_score = round(precision_score(y_test, y_pred), 4)
        print("accuracy_score: %.2f; \nf1_score: %.2f; \nauc_score: %.2f; \nprecision_score: %.2f;" % (
            acc_score, F1_score, auc_score, prec_score
        ))
        # fpr, tpr, threshold = roc_curve(y_test, y_pred)
        # import matplotlib.pyplot as plt
        # plt.title('ROC')
        # plt.plot(fpr, tpr, 'b', label='auc = %.2f' % auc_score)
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.show()
        return acc_score, F1_score, auc_score, prec_score
    else:
        pass


