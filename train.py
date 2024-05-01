import numpy as np
from dataset import Fashion_MNIST_Dataset, DataLoader, random_split
from model import Fashion_MNIST_Model
from utils import Softmax, SGD, ExpDecayLR
import pickle
from tqdm import tqdm



def train(model, train_loader, valid_loader, optimizer, sheduler, epoch):
    train_loss, valid_loss = [], []
    valid_acc = []
    best_acc = 0.0
    best_train_acc = 0.0
    
    for cur_epoch in tqdm(range(epoch)):
        cur_train_loss = 0.0
        cur_train_acc = 0.0
        cur_valid_loss = 0.0
        cur_valid_acc = 0.0
        data_len = 0

        if sheduler is not None:
            optimizer.set_multiplier(sheduler.get_multiplier(cur_epoch))

        train_pred, train_label = [], []
        for x, y in train_loader:
            cur_len = y.shape[0]
            data_len += cur_len

            pred = model.forward(x)
            train_pred.append(pred)
            train_label.append(y)
            loss = Softmax.loss(pred, y)
            loss_grad = Softmax.gradient(pred, y)
            model.backward(loss_grad)
            optimizer.update()

            cur_train_loss += loss
        train_pred = np.concatenate(train_pred)
        train_label = np.concatenate(train_label)

        cur_train_loss = Softmax.loss(train_pred, train_label)
        cur_train_acc = (np.argmax(train_pred, axis=1) == train_label).mean()

        valid_pred, valid_label = [], []
        for x, y in valid_loader:
            valid_pred.append(model.forward(x))
            valid_label.append(y)
        valid_pred = np.concatenate(valid_pred)
        valid_label = np.concatenate(valid_label)

        cur_valid_loss = Softmax.loss(valid_pred, valid_label)
        cur_valid_acc = (np.argmax(valid_pred, axis=1) == valid_label).mean()

        train_loss.append(cur_train_loss)
        valid_loss.append(cur_valid_loss)
        valid_acc.append(cur_valid_acc)

        print(
            f'Epoch: {cur_epoch + 1} Training Loss: {cur_train_loss:.5f} ' +
            f'Validation Loss: {cur_valid_loss:.5f} ' +
            f'Validation Accuracy: {cur_valid_acc:.5f}'
        )

        if cur_valid_acc > best_acc:
            best_acc = cur_valid_acc
            best_train_acc = cur_train_acc
            with open('./output/model.pickle', 'wb') as f:
                pickle.dump(model, f)

    return best_acc, best_train_acc, train_loss, valid_loss, valid_acc


if __name__ == '__main__':
    np.random.seed(1234)
    data_path = './data/'
    train_dataset, valid_dataset = random_split(Fashion_MNIST_Dataset(data_path, 'train'), (0.9, 0.1))
    train_loader = DataLoader(train_dataset, 64, True)
    valid_loader = DataLoader(valid_dataset, 128, True)

    lrs = [0.0001, 0.0005, 0.001]
    hidden_sizes = [64, 256, 576, 784]
    l2s = [0, 0.0001, 0.001]

    best_acc = 0.0
    best_train_acc = 0.0
    best_params = {}

    best_valid_acc_list = []
    best_valid_loss_list = []
    best_train_loss_list = []

    for lr in lrs:
        for hidden_size in hidden_sizes:
            for l2 in l2s:
                params = {'lr': lr, 'hidden_size': hidden_size, 'l2': l2}
                MNIST_model = Fashion_MNIST_Model(hidden_size, 10)
                optimizer = SGD(MNIST_model, lr=lr, l2=l2)
                sheduler = ExpDecayLR(1, 0.99)
                valid_acc, train_acc, train_loss_list, valid_loss_list, valid_acc_list = train(MNIST_model, train_loader, valid_loader, optimizer, sheduler, 100)
                print(params)
                print(f'Train Accuarcy: {train_acc} Validation Accuracy: {valid_acc}')

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_train_acc = train_acc
                    best_params = params

                    with open('./output/model.pickle', 'rb') as f:
                        curr_model = pickle.load(f)
                    curr_model.hyper_params = params
                    curr_model.train_loss_list = train_loss_list
                    curr_model.valid_loss_list = valid_loss_list
                    curr_model.valid_acc_list = valid_acc_list

                    with open('./output/best_model.pickle', 'wb') as f:
                        pickle.dump(curr_model, f)


    print(params)
    print(f'Train Accuarcy: {train_acc} Validation Accuracy: {valid_acc}')

    print('Best params:')
    print(best_params)
    print(f'Best Validation Accuracy: {best_acc} Train Accuracy: {best_train_acc}')