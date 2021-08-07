from numpy.core.numeric import outer
import torch
from torch.optim import optimizer
import torchaudio
import torch.nn as nn
import EmotionSpeechDataset 
from torch.utils.data import Dataset, DataLoader
import CNN_2D_Model as md
import torch.optim as opt

BATCH_SIZE = 16
NUM_WORKERS = 1
EPOCHS = 100
LEARNING_RATE = 0.001

train_dataset = EmotionSpeechDataset.EmotionSpeechDataset(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_train.csv")
val_dataset = EmotionSpeechDataset.EmotionSpeechDataset(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_validation.csv")
test_dataset = EmotionSpeechDataset.EmotionSpeechDataset(r"C:\Users\psiml\Desktop\PSIML_projekat\Ravdess_test.csv")

train_dataloader = DataLoader(dataset = train_dataset,
              batch_size = BATCH_SIZE,
              shuffle = True,
              num_workers = NUM_WORKERS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = md.CNN_2d_Model()
optimizer = opt.Adam(model.parameters(), lr = LEARNING_RATE, betas = (0.9,0.999))
loss_func = nn.CrossEntropyLoss()

# training loop
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)

        loss = loss_func(output, y)

        loss.backward()
        output.step()

    print(f'epoch: {epoch} loss: {loss}')
