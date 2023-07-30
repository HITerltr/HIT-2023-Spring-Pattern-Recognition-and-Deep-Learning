import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import models
import datasets


class Trainer:
    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = models.UNet().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters())
        self.loss_func = nn.BCELoss()
        self.loader = DataLoader(datasets.Datasets(path), batch_size=4, shuffle=True, num_workers=4)

        self.load_model()

    def load_model(self):
        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(self.model))
            print(f"Loaded {self.model}!")
        else:
            print("No Param!")

    def save_model(self, epoch, loss):
        torch.save(self.net.state_dict(), self.model)
        if epoch % 50 == 0:
            model_copy_path = self.model_copy.format(epoch, loss)
            torch.save(self.net.state_dict(), model_copy_path)
            print(f"{model_copy_path} is saved!")

    def save_image(self, inputs, outputs, labels, epoch):
        x = inputs[0].cpu()
        x_ = outputs[0].cpu()
        y = labels[0].cpu()
        img = torch.stack([x, x_, y], 0)
        save_image(img, os.path.join(self.img_save_path, f"{epoch}.png"))

    def train(self, stop_value):
        writer = SummaryWriter()

        for epoch in range(1, stop_value + 1):
            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}", ascii=True)
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.opt.step()

                self.save_image(inputs, outputs, labels, epoch)

                writer.add_scalar("Loss", loss, epoch)
                progress_bar.set_postfix({"Loss": loss.item()})

            self.save_model(epoch, loss)

        writer.close()


if __name__ == '__main__':
    t = Trainer(r"./DRIVE/training", r'./model.plt', r'./model_{}_{}.plt', img_save_path=r'./train_img')
    t.train(300)
