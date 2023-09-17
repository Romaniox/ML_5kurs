import os
import numpy as np
import cv2
import torch
import enum


class ModelType(enum.Enum):
    JIT = 1
    PT = 2


def main():
    imgs_path = os.path.join(os.getcwd(), 'test_data')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_name = 'exp13_best'
    model_type = ModelType.JIT

    match model_type:
        case ModelType.JIT:
            model = torch.jit.load(os.path.join(os.getcwd(), exp_name, 'model.pt'))
        case ModelType.PT:
            from train import CNN
            model = torch.load(os.path.join(os.getcwd(), exp_name, 'model.pth'))
        case _:
            raise ValueError('Invalid model type')

    model.eval()
    model.to(device)

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())

    for img_name in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (28, 28))
        img = img.astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        img = img.to(device)

        with torch.no_grad():
            output = model(img)
            prob = torch.nn.functional.softmax(output, dim=1)
            predicted = torch.max(prob.data, 1)[1]

            print(f'{img_name} is {predicted.item()} with probability {prob[0][predicted].item() * 100:.2f}%')


if __name__ == '__main__':
    main()
