import torch
import numpy as np
import timeit
# import matplotlib.pyplot as plt
from models.dp import ResNet50DP
from configurations import configResnetDP


def train(model):
    model.train(True)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(configResnetDP['batch_size']) \
                           .random_(0, configResnetDP['num_classes']) \
                           .view(configResnetDP['batch_size'], 1)

    for _ in range(configResnetDP['num_batches']):
        # generate random inputs and labels
        inputs = torch.randn(configResnetDP['batch_size'], 3,
                             configResnetDP['image_w'],
                             configResnetDP['image_h'])

        labels = torch.zeros(configResnetDP['batch_size'],
                             configResnetDP['num_classes']) \
                      .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


def main():
    # plt.switch_backend('Agg')

    num_repeat = 10

    stmt = "train(model)"

    setup = "model = ResNet50DP()"
    # globals arg is only available in Python 3. In Python 2, use the following
    # import __builtin__
    # __builtin__.__dict__.update(locals())
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

    setup = "import torchvision.models as models;" + \
            "model = models.resnet50(num_classes=configResnetDP['num_classes']).to('cuda:0')"

    rn_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)
    print(f'Single-GPU: {rn_mean}, {rn_std},\n DataParallel: {mp_mean}, {mp_std}')

    # plot([mp_mean, rn_mean],
    #      [mp_std, rn_std],
    #      ['Model Parallel', 'Single GPU'],
    #      'mp_vs_rn.png')



if __name__ == '__main__':
    main()
