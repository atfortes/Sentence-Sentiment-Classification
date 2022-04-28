import torch
import numpy as np
from test import test
from tqdm.auto import tqdm


def train(model, train_set, val_set, criterion, optimizer, epochs, batch_size, patience):
    best_accu, es_trigger = 0, 0
    loss_hist, accu_hist = [], []
    for epoch in range(epochs):
        correct = 0
        epoch_loss_hist = []
        model.train()
        for batch in tqdm(train_set):
            output, _ = model(*batch.text)
            train_y = batch.label

            optimizer.zero_grad()
            loss = criterion(output, train_y)
            epoch_loss_hist.append(loss.item())
            loss.backward()
            optimizer.step()

            correct += int((torch.max(output.data, 1)[1] == train_y).sum())

        val_accu = test(model, val_set, batch_size)
        loss_hist.append(np.mean(epoch_loss_hist))
        accu_hist.append(correct / len(train_set) / batch_size)
        print(f'epoch: {epoch+1} \t avg_loss={loss_hist[-1]:.4f} \t avg_acc={accu_hist[-1]:.4f} \t val_acc={val_accu:.4f}')

        es_trigger += 1
        if val_accu >= best_accu:
            best_accu = val_accu
            es_trigger = 0
        
        if es_trigger >= patience:
            print('early stopping...')
            break

    return loss_hist, accu_hist
