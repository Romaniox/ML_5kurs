import torch


def main():
    exp_name = 'exp12'
    model = torch.load(rf'D:\PyTorch\ml\CNN\{exp_name}\model.pth')

    traced_model = torch.jit.script(model)
    traced_model.save(rf'D:\PyTorch\ml\CNN\{exp_name}\model.pt')
    print('Finished')


if __name__ == '__main__':
    main()
