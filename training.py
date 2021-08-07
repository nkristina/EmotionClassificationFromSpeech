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
LEARNING_RATE = 0.00001
PATH = r"C:\Users\psiml\Desktop\PSIML_projekat\one_classr2.pt"

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

    spec = val_dataset[5][0]
    spec = torch.squeeze(spec,0)
    spec = spec.numpy()
    librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='mel') 
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    model = md.CNN_2d_Model()
    model.cuda()
    optimizer = opt.Adam(model.parameters(), lr = LEARNING_RATE, betas = (0.9,0.999))
    loss_func = nn.CrossEntropyLoss()

    summary_writer = SummaryWriter()
    metrics = defaultdict(list)

    best_acc_train = 0
    best_acc_val = 0
    # training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for i,(x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)

            loss = loss_func(output, y)

            loss.backward()
            optimizer.step()

            # statistics
            _, preds = torch.max(output, 1)
            running_loss += loss.item() * x.size(0)
            running_corrects += torch.sum(preds == y.data).item()

            #print(f'TRAINING: epoch: {epoch} iteration: {i} loss: {loss}')
            summary_writer.add_scalar("Traning loss", loss, i)
    
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = float(running_corrects) / len(train_dataset)
        metrics["test_loss"].append(epoch_loss)
        metrics["test_acc"].append(epoch_acc)

        summary_writer.add_scalar("Traning loss per epoch", epoch_loss, epoch)
        summary_writer.add_scalar("Traning acc per epoch", epoch_acc, epoch)
        
        print(f'Epoch: {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
        # deep copy the model
        if epoch_acc > best_acc_train:
            best_acc_train = epoch_acc

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for i, (x,y) in enumerate(val_dataloader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = loss_func(output, y)

            # statistics
            _, preds = torch.max(output, 1)
            running_loss += loss.item() * x.size(0)
            running_corrects += torch.sum(preds == y.data).item()

            print(f'VALIDATION: epoch: {epoch} loss: {loss}')
            summary_writer.add_scalar("Validation loss", loss, i)

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = float(running_corrects) / len(val_dataset)
        metrics["val_loss"].append(epoch_loss)
        metrics["val_acc"].append(epoch_acc)

        # deep copy the model
        if epoch_acc > best_acc_val:
            best_acc_val = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), PATH)

        summary_writer.add_scalar("Validation loss per epoch", epoch_loss, epoch)
        summary_writer.add_scalar("Validation acc per epoch", epoch_acc, epoch)

        print(f'Epoch: {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


    plt.plot(metrics["train_loss"])
    plt.title('Trening Loss funkcija po epohama')
    plt.savefig("loss_train.png")

    plt.plot(metrics["val_loss"])
    plt.title('Validacion loss funkcija po epohama')
    plt.savefig("loss_valid.png")

    plt.plot(metrics["train_acc"])
    plt.title('Trening acc po epohama')
    plt.savefig("acc_train.png")

    plt.plot(metrics["val_acc"])
    plt.title('Validacion acc po epohama')
    plt.savefig("acc_tvalid.png")