import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import numpy as np
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from skimage import io
import quality_metrics

batch_size = 16
epochs = 50
scaling_factor = 4
lr = 1e-4
device = torch.device('cuda:0')
ucm_dataset = dataset.UCMercedLandUseDataset(root_dir  = "/u/36/umashaa1/unix/Documents/cv4rs/UCMerced_LandUse")
train_set, val_set, test_set = torch.utils.data.random_split(ucm_dataset, [1470, 210, 420])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
validation_loader = torch.utils.data.DataLoader(val_set, batch_size=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=2, padding_mode='replicate') # padding mode same as original Caffe code
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2, padding_mode='replicate')
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR


model = SRCNN().to(device)
print(model)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# loss function
criterion = nn.MSELoss()

#####################
### TRAINING LOOP ###
#####################
train_loss, val_loss = [], []
train_psnr, val_psnr = [], []
val_rmse, val_sre = [], []
start = time.time()
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    for b_idx, (lr_imgs,hr_imgs) in enumerate(train_loader):
        lr_imgs = lr_imgs.permute(0,3,1,2).float()
        hr_imgs = hr_imgs.permute(0,3,1,2).float()

        image_data = lr_imgs.to(device)
        label = hr_imgs.to(device)

        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        # calculate batch psnr (once every `batch_size` iterations)
        batch_psnr =  psnr(label, outputs)
        running_psnr += batch_psnr
    train_epoch_loss = running_loss/len(train_loader)
    train_epoch_psnr = running_psnr/int(len(train_loader)/batch_size)

    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_rmse = 0.0
    running_sre = 0.0
    running_rmse_bicubic = 0.0
    running_sre_bicubic = 0.0
    with torch.no_grad():
        for bi, (lr_imgs,hr_imgs) in enumerate(validation_loader):
            lr_imgs = lr_imgs.permute(0,3,1,2).float()
            hr_imgs = hr_imgs.permute(0,3,1,2).float()

            image_data = lr_imgs.to(device)
            label = hr_imgs.to(device)

            outputs = model(image_data)
            loss = criterion(outputs, label)

            if epoch == 0:
                img_cpu, outputs_cpu = np.transpose(np.squeeze(label.cpu().detach().numpy(),axis=0), axes=(1,2,0)), np.transpose(np.squeeze(image_data.cpu().detach().numpy(),axis=0), axes=(1,2,0))
                val_rmse_img = quality_metrics.rmse(img_cpu, outputs_cpu)
                running_rmse_bicubic += val_rmse_img
                val_sre_img = quality_metrics.sre(img_cpu, outputs_cpu)
                running_sre_bicubic += val_sre_img

            # add loss of each item (total items in a batch = batch size)
            running_loss += loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr
            img_cpu, outputs_cpu = np.transpose(np.squeeze(label.cpu().detach().numpy(),axis=0), axes=(1,2,0)), np.transpose(np.squeeze(outputs.cpu().detach().numpy(),axis=0), axes=(1,2,0))
            val_rmse_img = quality_metrics.rmse(img_cpu, outputs_cpu)
            running_rmse += val_rmse_img
            val_sre_img = quality_metrics.sre(img_cpu, outputs_cpu)
            running_sre += val_sre_img
        outputs = outputs.cpu()

        #### ENABLE TO VISUALISE TRAINING
        #io.imshow(np.transpose(np.squeeze(np.concatenate((label.cpu().numpy(), image_data.cpu().numpy(), outputs.cpu().numpy()), axis=3),axis=0), axes=(1,2,0)))
        #plt.show()
        save_image(torch.cat((label.cpu(), image_data.cpu(), outputs.cpu()), axis=3), f"images/orig_bicubic_srcnn_{epoch}.png")
    if epoch == 0:
        print("Val set Bicubic rmse, sre ",running_rmse_bicubic/len(validation_loader),running_sre_bicubic/len(validation_loader))
    val_epoch_loss = running_loss/len(validation_loader)
    val_epoch_psnr = running_psnr/int(len(validation_loader)/batch_size)
    val_epoch_rmse = running_rmse/len(validation_loader)
    val_epoch_sre = running_sre/len(validation_loader)
    train_loss.append(train_epoch_loss)
    train_psnr.append(train_epoch_psnr)
    val_loss.append(val_epoch_loss)
    val_psnr.append(val_epoch_psnr)
    val_rmse.append(val_epoch_rmse)
    val_sre.append(val_epoch_sre)
    print(train_epoch_loss, train_epoch_psnr, val_epoch_loss, val_epoch_psnr)
    print(val_epoch_rmse,val_epoch_sre)
    torch.save({'epoch': epoch,'model': model,'optimizer': optimizer},'checkpoint_srgan.pth.tar')
end = time.time()
print(f"Finished training in: {((end-start)/60):.3f} minutes")


#####################
### TESTING LOOP ####
#####################
running_rmse_test = 0.0
running_sre_test = 0.0
running_rmse_bicubic_test = 0.0
running_sre_bicubic_test = 0.0
with torch.no_grad():
    for bi, (lr_imgs,hr_imgs) in enumerate(test_loader):
        lr_imgs = lr_imgs.permute(0,3,1,2).float()
        hr_imgs = hr_imgs.permute(0,3,1,2).float()

        image_data = lr_imgs.to(device)
        label = hr_imgs.to(device)

        outputs = model(image_data)


        img_cpu, outputs_cpu = np.transpose(np.squeeze(label.cpu().detach().numpy(),axis=0), axes=(1,2,0)), np.transpose(np.squeeze(image_data.cpu().detach().numpy(),axis=0), axes=(1,2,0))
        val_rmse_img = quality_metrics.rmse(img_cpu, outputs_cpu)
        running_rmse_bicubic_test += val_rmse_img
        val_sre_img = quality_metrics.sre(img_cpu, outputs_cpu)
        running_sre_bicubic_test += val_sre_img

        img_cpu, outputs_cpu = np.transpose(np.squeeze(label.cpu().detach().numpy(),axis=0), axes=(1,2,0)), np.transpose(np.squeeze(outputs.cpu().detach().numpy(),axis=0), axes=(1,2,0))
        val_rmse_img = quality_metrics.rmse(img_cpu, outputs_cpu)
        running_rmse_test += val_rmse_img
        val_sre_img = quality_metrics.sre(img_cpu, outputs_cpu)
        running_sre_test += val_sre_img

        save_image(torch.cat((label.cpu(), image_data.cpu(), outputs.cpu()), axis=3), f"images/test_{bi}.png")

print("Bicubic rmse, sre : ",running_rmse_bicubic_test/len(test_loader),running_sre_bicubic_test/len(test_loader))
test_epoch_rmse = running_rmse_test/len(test_loader)
test_epoch_sre = running_sre_test/len(test_loader)
print("SRCNN rmse, sre : ",test_epoch_rmse,test_epoch_sre)
