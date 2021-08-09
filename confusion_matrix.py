from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def conf_matrix_metrics(y_true, y_pred, labels):
    
    c_matrix = confusion_matrix(y_true, y_pred)

    '''df_cm = pd.DataFrame(c_matrix, index=labels, columns=labels)

    plt.figure()
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('conf_mat_WD.png')'''

    return c_matrix

def plot_conf_mat(c_matrix, name, labels):
    df_cm = pd.DataFrame(c_matrix, index=labels, columns=labels)

    plt.figure()
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name)