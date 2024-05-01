import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from model import Linear
import os



if __name__ == '__main__':
    with open('./output/best_model.pickle', 'rb') as f:
        best_model = pickle.load(f)

    sns.set(style="whitegrid")

    # 可视化best_model训练过程
    train_loss_list = best_model.train_loss_list
    valid_loss_list = best_model.valid_loss_list
    valid_acc_list = best_model.valid_acc_list

    epochs = range(1, len(valid_loss_list) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss_list, 'b', label='Training loss')
    plt.plot(epochs, valid_loss_list, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/Users/wangdixuan/Documents/研一下/神经网络和深度学习/hw1/figures/loss.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, valid_acc_list, 'b', label='Validation accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/Users/wangdixuan/Documents/研一下/神经网络和深度学习/hw1/figures/accuracy.png')
    plt.show()


    # 权重可视化
    hidden_size = best_model.hyper_params['hidden_size']
    print(hidden_size)
    params, _ = best_model.get_params_and_grads()

    weight1 = params[:28*28*hidden_size].reshape(28*28, hidden_size)
    bias1 = params[28*28*hidden_size: 28*28*hidden_size+hidden_size].reshape(hidden_size, 1)
    weight2 = params[28*28*hidden_size+hidden_size: -10].reshape(hidden_size, 10)
    bias2 = params[-10:].reshape(10, 1)

    def visualize_weights(weights, save_dir):
        for i, W in enumerate(weights):
            input_layer, output_layer = W.shape # (784, hidden_size)
            print(input_layer, output_layer)
            num_rows = int(output_layer ** 0.5) # 16
            num_cols = output_layer // num_rows + 1 # 17
            plt.figure(figsize=(12, 8))

            for j in range(output_layer):
                plt.subplot(num_rows, num_cols, j+1)
                plt.imshow(W[:,j].reshape(int(input_layer ** 0.5), -1))
                plt.axis('off')

            plt.savefig(os.path.join(save_dir, f'Weights{i+1}.png'))
            plt.close()

    visualize_weights([weight1, weight2], '/Users/wangdixuan/Documents/研一下/神经网络和深度学习/hw1/figures')
