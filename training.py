from numpy.core.numeric import outer
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import optimizer
import torchaudio
import torch.nn as nn
import EmotionSpeechDataset 
from torch.utils.data import Dataset, DataLoader
import CNN_2D_Model as md
import torch.optim as opt
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import librosa


BATCH_SIZE = 64
NUM_WORKERS = 5
EPOCHS = 100
LEARNING_RATE = 0.0001
PATH = r"C:\Users\psiml\Desktop\PSIML_projekat\Models\NoBN_2c_18_4_2.pt"

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
    optimizer = opt.Adam(model.parameters(), lr = LEARNING_RATE, betas = (0.9,0.999))
    loss_func = nn.CrossEntropyLoss()

    summary_writer = SummaryWriter()
    metrics = defaultdict(list)

    best_acc = 0
    stop_count = 0
    stop = False
    # training loop
    for epoch in range(EPOCHS):
        for state in ['train', 'valid']:

            if state == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i,(x, y) in enumerate(dataloaders[state]):
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(state == 'train'):
                    output = model(x)
                    _, preds = torch.max(output, 1)
                    loss = loss_func(output, y)

                if state == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * x.size(0)
                running_corrects += torch.sum(preds == y.data).item()

                summary_writer.add_scalar(str(state) + " loss", loss, i)
            
            epoch_loss = running_loss / len(datasets[state])
            epoch_acc = float(running_corrects) / len(datasets[state])
            metrics[state + "_loss"].append(epoch_loss)
            metrics[state + "_acc"].append(epoch_acc)

            summary_writer.add_scalar(state + " loss per epoch", epoch_loss, epoch)
            summary_writer.add_scalar(state + " acc per epoch", epoch_acc, epoch)
        
            print(f'Epoch: {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} State: {state}')
            
            if state == 'valid':
                if epoch_acc > 0.05*best_acc:
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

        if stop:
            break

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