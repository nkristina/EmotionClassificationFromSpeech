from numpy.core.numeric import outer
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import optimizer
import torchaudio
import torch.nn as nn
import EmotionSpeechDataset 
from torch.utils.data import Dataset, DataLoader
import CNN_2D_Model_7clas as md
import torch.optim as opt
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import confusion_matrix

BATCH_SIZE = 1
NUM_WORKERS = 5
EPOCHS = 100
LEARNING_RATE = 0.0001
PATH = r"C:\Users\psiml\Desktop\PSIML_projekat\Models\NoBN_WD_14c_20_2_2.pt"

target = []
predicted = []
emotions = {0: 'female_neutral', 1: 'female_happy', 2: 'female_sad', 3: 'female_angry', 4: 'female_fear', 5: 'female_disgust', 6: 'female_surprise',
            7: 'male_neutral', 8: 'male_happy', 9: 'male_sad', 10: 'male_angry', 11: 'male_fear', 12: 'male_disgust', 13: 'male_surprise'}

if __name__ == '__main__':

    test_dataset = EmotionSpeechDataset.EmotionSpeechDataset(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_validation.csv")

    test_dataloader = DataLoader(dataset = test_dataset,
                batch_size = BATCH_SIZE,
                shuffle = False,
                num_workers = NUM_WORKERS)

    device = torch.device("cuda")
    loss_func = nn.CrossEntropyLoss()

    model = md.CNN_2d_Model()
    model.load_state_dict(torch.load(PATH))
    model.cuda()
    model.eval()

    running_loss = 0
    running_corrects = 0
    topK_corrects = 0

    for i,(x, y) in enumerate(test_dataloader):
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(False):
            output = model(x)
            _, preds = torch.max(output, 1)
            _, topK = torch.topk(output, 3, 1)
            loss = loss_func(output, y)

        running_loss += loss.item() * x.size(0)
        running_corrects += torch.sum(preds == y.data).item()

        for i in range(len(preds)):
            topK_corrects += y.data[i] in topK[i]

        predicted.extend(preds.tolist())
        target.extend(y.tolist())
    
    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = float(running_corrects) / len(test_dataset)
    epoch_topK = float(topK_corrects) / len(test_dataset)

    c_matrix = confusion_matrix.conf_matrix_metrics(target, predicted, emotions.values())
    confusion_matrix.plot_conf_mat(c_matrix, 'conf_mat_best_14c_val.png', emotions.values())

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Top3: {epoch_topK:4f}')

    predicted_df = pd.DataFrame(predicted, columns = ['Predicted'])
    target_df = pd.DataFrame(target, columns = ['Target'])

    predicted_df.Predicted.replace(emotions, inplace=True)
    target_df.Target.replace(emotions, inplace=True)

    Test_df = pd.concat([target_df, predicted_df], axis=1)
    Test_df.to_csv(r"C:\Users\psiml\Desktop\PSIML_projekat\Val.csv",index=False)