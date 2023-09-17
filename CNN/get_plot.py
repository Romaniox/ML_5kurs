import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv(r'D:\PyTorch\ml\CNN\exp1\metrics.csv')

    sns.set(style="whitegrid")

    # Создаем фигуру с двумя осями
    plt.figure(figsize=(12, 6))

    # Первый график - Loss
    plt.subplot(1, 2, 1)  # Один ряд, два столбца, первый график
    sns.lineplot(x='epoch', y='loss', data=data, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('График Loss')
    plt.legend()

    # Второй график - Accuracy
    plt.subplot(1, 2, 2)  # Один ряд, два столбца, второй график
    sns.lineplot(x='epoch', y='accuracy', data=data, label='Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('График Accuracy')
    plt.legend()

    # Регулируем расположение графиков
    plt.tight_layout()

    # Сохраняем график
    plt.savefig(r'D:\PyTorch\ml\CNN\exp1\metrics.png')


if __name__ == '__main__':
    main()
