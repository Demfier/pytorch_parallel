import torch
from models.dp import ToyModelDP
from configurations import configDP


def train():
    model = ToyModelDP()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=configDP['lr'])

    optimizer.zero_grad()
    outputs = model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to('cuda:1')
    loss_fn(outputs, labels).backward()
    optimizer.step()


if __name__ == '__main__':
    train()
