import numpy as np
import torch
import math
import imageio.v2 as imageio
import os
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.optim import lr_scheduler, AdamW
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn

matplotlib.use('Agg')

device = torch.device('cuda')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Dataset_Denoise(data.Dataset):
    def __init__(self, path, sigma=0.2):
        super(Dataset_Denoise, self).__init__()
        self.path = path
        self.sigma = sigma
        self.list_image = []
        for item in os.listdir(path):
            if '.png' in item:
                self.list_image.append(os.path.join(path, item))
        assert len(self.list_image) > 0

    def __getitem__(self, index):
        x = imageio.imread(self.list_image[index], pilmode='L').astype(np.float32).reshape(1, 28, 28)
        x = x / x.max()
        y = x + self.sigma * np.random.randn(*x.shape)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def __len__(self):
        return len(self.list_image)


class SimpleDenoiseNN(nn.Module):
    def __init__(self):
        super(SimpleDenoiseNN, self).__init__()
        self.input_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True),
        )
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True),
        )
        self.output_layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=(1, 1), bias=True),
        )

    def forward(self, x):
        x = self.input_layers(x)
        y = self.hidden_layers(x)
        y += x
        y = self.output_layers(y)
        return y


def cal_psnr(img1, img2):
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0: return float('inf')
    result = 20 * math.log10(img1.max() / math.sqrt(mse))
    return result


def main():
    mkdir('denoising')
    epoch_num = 4
    print_interval = 100
    validate_interval = 200
    savemodel_interval = 2000
    train_loader = DataLoader(Dataset_Denoise('minist_dataset/train'), batch_size=4, num_workers=4, shuffle=True, drop_last=True, pin_memory=False)
    val_loader = DataLoader(Dataset_Denoise('minist_dataset/val'), batch_size=1, num_workers=1, shuffle=False, drop_last=False, pin_memory=False)
    net = SimpleDenoiseNN().to(device)
    optim_params = []
    for k, v in net.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
    optimizer = AdamW(optim_params, lr=1e-3, weight_decay=0.)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num * train_loader.__len__(), eta_min=1e-6)
    lossfn = nn.MSELoss()

    current_step = 0
    loss_idx, loss_value, psnr_idx, psnr_value = [], [], [], []
    for epoch in range(epoch_num):
        for _, train_data in enumerate(train_loader):
            x, y = train_data[0].to(device), train_data[1].to(device)
            optimizer.zero_grad()
            xhat = net(y)
            loss = lossfn(x, xhat)
            current_step += 1
            loss.backward()
            optimizer.step()
            scheduler.step()

            if current_step % print_interval == 0:
                print("iterNum {:8,d} | loss {:.3e}".format(current_step, loss.item()))
                loss_idx.append(current_step)
                loss_value.append(loss.item())

            if current_step % savemodel_interval == 0:
                state_dict = net.state_dict()
                for key, param in state_dict.items():
                    state_dict[key] = param
                mkdir('denoising/model')
                torch.save(state_dict, 'denoising/model/net_{:0>6}.pth'.format(current_step))

            if current_step % validate_interval == 0:
                psnr_list = []
                for val_idx, val_data in enumerate(val_loader):
                    net.eval()
                    xval, yval = val_data[0].to(device), val_data[1].to(device)
                    with torch.no_grad():
                        xvalhat = net(yval)

                    mkdir('denoising/images/{:0>3}'.format(val_idx))
                    if current_step == validate_interval:
                        imageio.imwrite('denoising/images/{:0>3}/{:0>8}_gt.tif'.format(val_idx, current_step), xval.squeeze().cpu().numpy().astype(np.float32))
                        imageio.imwrite('denoising/images/{:0>3}/{:0>8}_noisy.tif'.format(val_idx, current_step), yval.squeeze().cpu().numpy().astype(np.float32))
                    imageio.imwrite('denoising/images/{:0>3}/{:0>8}_denoised.tif'.format(val_idx, current_step), xvalhat.squeeze().cpu().numpy().astype(np.float32))

                    psnr = cal_psnr(xval, xvalhat)
                    psnr_list.append(psnr)
                    net.train()

                print("iterNum {:8,d} | validation PSNR {:<5.2f} dB".format(current_step, np.mean(psnr_list)))
                psnr_idx.append(current_step)
                psnr_value.append(np.mean(psnr_list))

                fig = plt.figure()

                plt.subplot(121)
                plt.plot(loss_idx, loss_value, color='black', marker='o')
                plt.title('loss')

                plt.subplot(122)
                plt.plot(psnr_idx, psnr_value, color='black', marker='o')
                plt.title('psnr')

                # savefig
                fig.set_size_inches(20, 7)
                try:
                    fig.savefig('denoising/plot.png', format='png', transparent=False, dpi=100, pad_inches=0)
                except OSError:  # builtins error
                    pass
                plt.close()


if __name__ == '__main__':
    main()
