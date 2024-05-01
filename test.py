import numpy as np
from dataset import Fashion_MNIST_Dataset, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(confusion_mat, class_names):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontweight='bold', fontsize=16, pad=20)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30)
    plt.yticks(tick_marks, class_names)

    thresh = confusion_mat.max() / 2.
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            plt.text(j, i, format(confusion_mat[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_mat[i, j] > thresh else "black")

    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/wangdixuan/Documents/研一下/神经网络和深度学习/hw1/figures/confusion_matrix')
    plt.show()

    

if __name__ == '__main__':
    data_path = './data/'
    test_dataset = Fashion_MNIST_Dataset(data_path, 'test')
    test_loader = DataLoader(test_dataset, 128, False)
    # print(test_dataset.labels)

    with open('./output/best_model.pickle', 'rb') as f:
        best_model = pickle.load(f)
        test_pred, test_label = [], []

        for x, y in test_loader:
            test_pred.append(best_model.forward(x))
            test_label.append(y)
        test_pred = np.concatenate(test_pred)
        test_label = np.concatenate(test_label)
        test_acc = (np.argmax(test_pred, axis=1) == test_label).mean()
        print(f'Test Accuracy: {test_acc:.5f}')

    # 可视化混淆矩阵
    conf_mat = confusion_matrix(test_label, np.argmax(test_pred, axis=1))
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plot_confusion_matrix(conf_mat, class_names)