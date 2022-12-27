import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,recall_score, auc, roc_curve,precision_score,accuracy_score,f1_score,precision_recall_curve
import matplotlib.pyplot as plt


def acc_pre_recal_f1(y_true, y_pred):
    cnf_matrix = pd.DataFrame(data=confusion_matrix(y_true, y_pred),
                          columns=['预测结果为正例','预测结果为反例'],index=['真实样本为正例','真实样本为反例'])
    print("数据集上的混淆矩阵：")
    print(cnf_matrix)
    print("-"*40)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print('模型预测的正确率:',acc)
    print('模型的精准率：',precision)
    print('模型的召回率：',recall)
    print('模型的f1值：',f1)
    return acc,precision,recall,f1

def pr_cover(train_true, train_pred_prob, test_true, test_preb_prob):
    train_precision, train_recall, _ = precision_recall_curve(train_true, train_pred_prob)
    test_precision, test_recall, _ = precision_recall_curve(test_true, test_preb_prob)
    plt.figure(figsize=(8,5))  
    plt.plot(train_recall, train_precision,color = 'r', linestyle='-',label='训练集P-R曲线')
    plt.plot(test_recall, test_precision,color = 'b', linestyle=':',label='测试集P-R曲线')
    plt.xlabel('Recall',fontsize=12)
    plt.ylabel('Precision',fontsize=12)
    plt.legend(fontsize=16)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xticks( fontsize=12)
    plt.yticks( fontsize=12)
    plt.show()

def roc_ks_cover(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ar = 2*roc_auc-1
    gini = ar
    
    print("roc取线: (auc值{0})".format(roc_auc))
    plt.figure(figsize=(8,5))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks( fontsize=12)
    plt.yticks( fontsize=12)
    plt.xlabel('FPR',fontsize=14)
    plt.ylabel('TPR',fontsize=14)
    plt.title('ROC',fontsize=14)
    plt.legend(loc="lower right",fontsize=14)   
    plt.show()

    print("-"*60)
    ks = max(tpr - fpr)
    print("ks曲线: (ks值{0})".format(ks))
    plt.figure(figsize=(8,5))
    plt.plot(np.linspace(0,1,len(tpr)),tpr,'--',color='red', label='positive-ks')
    plt.plot(np.linspace(0,1,len(tpr)),fpr,':',color='blue', label='negative-ks')
    plt.plot(np.linspace(0,1,len(tpr)),tpr - fpr,'-',color='green')
    plt.grid()
    plt.xticks( fontsize=12)
    plt.yticks( fontsize=12)
    plt.xlabel('% of population',fontsize=14)
    plt.ylabel('% of total Good/Bad',fontsize=14)
    plt.legend(fontsize=14)
    plt.show()

    return roc_auc,ks