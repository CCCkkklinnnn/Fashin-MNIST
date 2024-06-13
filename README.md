# Pytorch实战任务1

**Kaggle Fashin-MNIST数据集实现分类任务**

**任务流程：**

1. **导入包**

2. **基本配置**（批次大小、学习率、epoch、gpu）

3. **Dataset 建立并 Load**
  - 3.1 建立自己的 Dataset 类
  - 3.2 读取 CSV 数据(train_df, test_df)
  - 3.3 实例化 Dataset 类(train_data, test_data)
  - 3.4 定义 DataLoader 类加载数据(train_loader, test_loader)
    
4.**构建网络**（初始化+前向传播CNN+FC）

5.**损失函数** CrossEntropy

6.**优化器** Adam

7.**训练**
  - 7.1训练模式（以下是对每个批次）
  - 7.2把数据放到gpu上
  - 7.3数据经过网络
  - 7.4计算loss
  - 7.5梯度清零
  - 7.6反向传播
  - 7.7优化参数

8.**评估**
  - 8.1评估模式
  - 8.2把数据放到gpu上
  - 8.3经过网络
  - 8.4计算loss
  - 8.5计算acc（要把结果放回cpu再转到数组）
