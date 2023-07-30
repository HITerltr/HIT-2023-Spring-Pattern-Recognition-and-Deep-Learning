import os
import torch
import datasets
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import models

class Test:
    def __init__(self, path, model_path, img_save_path):
        self.path = path
        self.model_path = model_path
        self.img_save_path = img_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = models.UNet().to(self.device)
        self.loss_func = nn.BCELoss()
        self.loader = DataLoader(datasets.Datasets(path), batch_size=2, shuffle=True, num_workers=4)
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path))
            print(f"Loaded {model_path}!")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}!")
        os.makedirs(img_save_path, exist_ok=True)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            for inputs, labels in self.loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.loss_func(outputs, labels)
                print(f"Batch Loss: {loss.item()}")

                for i in range(inputs.size(0)):
                    input_img = inputs[i]
                    output_img = outputs[i]
                    label_img = labels[i]
                    img = torch.stack([input_img, output_img, label_img], 0)
                    save_image(img.cpu(), os.path.join(self.img_save_path, f"test_{i}.png"))

if __name__ == '__main__':
    test = Test(r"./DRIVE/test", r'./model.plt', img_save_path=r'./test_results')
    test.test()
