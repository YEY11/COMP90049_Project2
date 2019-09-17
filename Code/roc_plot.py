## This program is used to plot the ROC curves of three machine learning 
## models based on the relevant roc result csv files exported from WEKA.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, auc

def roc_plot(rocfile_NB, rocfile_MLR, rocfile_RF):
    # Reading the ROC data
    roc_nb = pd.read_csv(rocfile_NB)
    roc_mlr = pd.read_csv(rocfile_MLR)
    roc_rf = pd.read_csv(rocfile_RF)

    # Get the sentiment class for axis label in plot
    start = rocfile_NB.find('/ROC') + 5
    end = rocfile_NB.find('_')
    sentiment = rocfile_NB[start:end]

    # True Positive Rate and False Positive Rate
    fpr_nb = roc_nb["'False Positive Rate'"]
    tpr_nb = roc_nb["'True Positive Rate'"]

    fpr_mlr = roc_mlr["'False Positive Rate'"]
    tpr_mlr = roc_mlr["'True Positive Rate'"]

    fpr_rf = roc_rf["'False Positive Rate'"]
    tpr_rf = roc_rf["'True Positive Rate'"]

    # AUC (Area under Curve)
    nb_area = auc(fpr_nb, tpr_nb)
    mlr_area = auc(fpr_mlr, tpr_mlr)
    rf_area = auc(fpr_rf, tpr_rf)

    # Plot ROC Curves
    plt.figure(sentiment)
    plt.plot([0, 1], [0, 1], color='g',linestyle='--')
    plt.plot(fpr_nb, tpr_nb, color='r', label='Naive Bayes (AUC = % 0.3f)' % nb_area)
    plt.plot(fpr_mlr, tpr_mlr, color='dodgerblue', label='Multinomial LR (AUC = % 0.3f)' % mlr_area)
    plt.plot(fpr_rf, tpr_rf, color='orange', label='Random Forest (AUC = % 0.3f)' % rf_area)
    plt.xlabel('False Positive Rate '+'('+sentiment+')')
    plt.ylabel('True Positive Rate '+'('+sentiment+')')
    plt.legend(loc="lower right")
    plt.show()

# Get the roc results of 3 machine learning models for ROC plotting
negative_NB = "../ROC/negative_NB.csv"
negative_MLR = "../ROC/negative_MLR.csv"
negative_RF = "../ROC/negative_RF.csv"

positive_NB = "../ROC/positive_NB.csv"
positive_MLR = "../ROC/positive_MLR.csv"
positive_RF = "../ROC/positive_RF.csv"

neutral_NB = "../ROC/neutral_NB.csv"
neutral_MLR = "../ROC/neutral_MLR.csv"
neutral_RF = "../ROC/neutral_RF.csv"

# Plot ROC curves for 3 sentiment classes
roc_plot(negative_NB, negative_MLR, negative_RF)
roc_plot(positive_NB, positive_MLR, positive_RF)
roc_plot(neutral_NB, neutral_MLR, neutral_RF)