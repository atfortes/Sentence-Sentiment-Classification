import torch


def test(model, test_set, batch_size):
    correct = 0
    model.eval()
    for batch in test_set:
        x, l = batch.text
        train_y = batch.label
        output, _ = model(x, l)
        correct += int((torch.max(output.data, 1)[1] == train_y).sum())
    return correct / len(test_set) / batch_size
