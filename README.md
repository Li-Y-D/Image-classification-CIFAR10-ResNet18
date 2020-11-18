# Image-classification-CIFAR10-ResNet18

## Auther: Yuandong Li
## Date: 2020/11/17
## Description: The images in the CIFAR10 data set are classified using the ResNet18 residual network with 90% accuracy. 

# 欢迎关注我的CSDN博客—————夏风喃喃

## 一.实验要求：
仿照示例，使用jupyter notebook+pytorch完成一个网络的训练。 要求：
1. 使用任意的网络结构（可以更改网络的层数、卷积尺寸、channel数等等）
2. 使用cifar10数据集
3. 最后的分类精度至少达到0.9

## 二.实验准备：

硬件条件：

独立显卡型号：NVIDIA GeForce 940MX × 1

独立显卡显存：2048 MB

软件条件：

语言环境：Python 3.7.2

实验工具：Jupyter Notebook

深度学习框架库：PyTorch

## 三.实验内容：

数据集: CIFAR10

网络模型: ResNet18(有所修改)

源代码及模型的描述均以注释形式展示

## 四.实验结果：

适用于CIFAR10的ResNet18网络模型结构：

![images1](https://github.com/Li-Y-D/Image-classification-CIFAR10-ResNet18/blob/main/images/%E6%88%AA%E5%9B%BE1.png)

训练过程，展示训练集和测试集精确度以及训练用时：

![images1](https://github.com/Li-Y-D/Image-classification-CIFAR10-ResNet18/blob/main/images/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202020-11-18%20093746.jpg)

测试集数据经已训练好的网络模型所得到的预测标签与真实标签的对比：

![images1](https://github.com/Li-Y-D/Image-classification-CIFAR10-ResNet18/blob/main/images/%E6%88%AA%E5%9B%BE2.png)
