# Neural-Netwrok-and-Deep-Learning
##文件说明
- preprocess.py：对Fashion-MNIST的四个数据集进行初步处理，将其转化为numpy格式保存方便后续使用。
- dataset.py：实现了基础的数据集类和数据迭代器以及对训练集的划分方法。
- model.py：实现了基础的线性层、激活函数以及适用的两层线性分类器，并且都有对应的参数更新和梯度计算、反向传播方法。其中允许自定义隐藏层大小以及激活函数类型。
- utils.py：实现了基础的负对数损失函数、SGD优化器和指数学习率下降策略。
- train.py：实现了训练pipeline，使用上述自定义的类统一进行数据集构建、模型构建和训练，最终保存不同参数组合下的best model。
- test.py：实现了测试集上的accuracy计算。
- visualize.py：实现了可视化功能，包括训练过程中在训练集和验证集上的loss曲线和验证集上的accuracy曲线，以及对训练好的模型网络参数的可视化。

##使用方法
- 执行python preprocess.py进行数据预处理（需保证data文件夹下存在Fashion-MNIST的四个数据集）。
- 执行python train.py进行网格搜索，得到在验证集上最优表现的模型best model，自动保存在output文件夹下。
- 执行`python test.py进行测试集上的accuracy计算，绘制混淆矩阵，评测模型质量。
- 执行python visu
