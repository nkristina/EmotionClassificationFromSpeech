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
import confusion_matrix


BATCH_SIZE = 64
NUM_WORKERS = 5
EPOCHS = 100
LEARNING_RATE = 0.0001
PATH = r"C:\Users\psiml\Desktop\PSIML_projekat\Models\NoBN_WD_14c_20_2_2.pt"

LR_MIN = 0.0001
LR_MAX = 0.001
STEP = (LR_MAX - LR_MIN) / 10

target = []
predicted = []
best_target = []
best_predicted = []
emotions = {0: 'female_neutral', 1: 'female_happy', 2: 'female_sad', 3: 'female_angry', 4: 'female_fear', 5: 'female_disgust', 6: 'female_surprise',
            7: 'male_neutral', 8: 'male_happy', 9: 'male_sad', 10: 'male_angry', 11: 'male_fear', 12: 'male_disgust', 13: 'male_surprise'}

if __name__ == '__main__':

    train_dataset = EmotionSpeechDataset.EmotionSpeechDataset(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_train.csv")
    val_dataset = EmotionSpeechDataset.EmotionSpeechDataset(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_validation.csv")
    test_dataset = EmotionSpeechDataset.EmotionSpeechDataset(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_test.csv")

    train_dataloader = DataLoader(dataset = train_dataset,
                batch_size = BATCH_SIZE,
                shuffle = True,
                num_workers = NUM_WORKERS)

    val_dataloader = DataLoader(dataset = val_dataset,
                batch_size = BATCH_SIZE,
                shuffle = True,
                num_workers = NUM_WORKERS)

    dataloaders = {'train': train_dataloader, 'valid': val_dataloader}
    datasets = {'train': train_dataset, 'valid': val_dataset}

    spec = train_dataset[50][0]
    spec = torch.squeeze(spec,0)
    spec = spec.numpy()
    #librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='mel') 
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    #plt.colorbar()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    model = md.CNN_2d_Model()
    model.cuda()
    
    optimizer = opt.AdamW(model.parameters(), lr = LEARNING_RATE, betas = (0.9,0.999), weight_decay=0.05)
    loss_func = nn.CrossEntropyLoss()

    summary_writer = SummaryWriter()
    metrics = defaultdict(list)

    best_acc = 0
    topK_corrects = 0
    stop_count = 0
    stop = False
    topK_corrects = 0
    
    # training loop
    for epoch in range(EPOCHS):

        '''if epoch <= 10:
            LEARNING_RATE = LR_MAX - epoch*STEP
        else:
            LEARNING_RATE = LR_MIN'''

        #optimizer = opt.AdamW(model.parameters(), lr = LEARNING_RATE, betas = (0.9,0.999), weight_decay=0.05)
        
        for state in ['train', 'valid']:

            if state == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            topK_corrects = 0

            predicted = []
            target = []

            for i,(x, y) in enumerate(dataloaders[state]):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(state == 'train'):
                    output = model(x)
                    norm = torch.nn.functional.softmax(output)
                    _, preds = torch.max(norm, 1)
                    _, topK = torch.topk(norm, 3, 1)
                    loss = loss_func(output, y)

                if state == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y.data).item()

                for i in range(len(preds)):
                    topK_corrects += y.data[i] in topK[i]

                if state == 'valid':
                    predicted.extend(preds.tolist())
                    target.extend(y.tolist())

                summary_writer.add_scalar(str(state) + " loss", loss, i)
            
            epoch_loss = running_loss / len(datasets[state])
            epoch_acc = float(running_corrects) / len(datasets[state])
            epoch_topK = float(topK_corrects) / len(datasets[state])
            metrics[state + "_loss"].append(epoch_loss)
            metrics[state + "_acc"].append(epoch_acc)

            summary_writer.add_scalar(state + " loss per epoch", epoch_loss, epoch)
            summary_writer.add_scalar(state + " acc per epoch", epoch_acc, epoch)
        
            print(f'Epoch: {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} State: {state}')
            
            if state == 'valid':
                if epoch_acc > best_acc:
                    stop_count = 0
                else:
                    stop_count += 1
                    if stop_count > 10:
                        stop = True

            # deep copy the model
            if state == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), PATH)
                TOP3 = epoch_topK
                best_predicted = predicted
                best_target = target
                predicted = []
                target = []

        if stop:
            break

    print(TOP3)
    
    c_matrix = confusion_matrix.conf_matrix_metrics(best_target, best_predicted, emotions.values())
    confusion_matrix.plot_conf_mat(c_matrix, 'conf_mat_best_14c_val.png', emotions.values())

    plt.figure(1)
    plt.plot(metrics["train_loss"], 'b', metrics["val_loss"], 'r')
    plt.title('Loss funkcija po epohama')
    plt.xlabel('Epoha')
    plt.ylabel('Loss funkcija')
    plt.savefig("loss.png")

    plt.figure(2)
    plt.plot(metrics["train_loss"], 'b', metrics["val_loss"], 'r')
    plt.title('Tacnost po epohama')
    plt.xlabel('Epoha')
    plt.ylabel('Tacnost funkcija')
    plt.savefig("acc.png")