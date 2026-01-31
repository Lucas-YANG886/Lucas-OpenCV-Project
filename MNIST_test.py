import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，保证结果可重复
torch.manual_seed(42)

# 1. 数据加载与预处理 ===========================================
def load_data(batch_size=64):
    """
    加载MNIST数据集并进行预处理
    """
    # 定义数据转换：转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像转换为Tensor，并自动归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])
    
    # 下载并加载训练集
    train_dataset = datasets.MNIST(
        root='./data',  # 数据存储路径
        train=True,     # 训练集
        download=True,  # 如果本地没有则下载
        transform=transform
    )
    
    # 下载并加载测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,    # 测试集
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 每批数据大小
        shuffle=True,           # 训练时打乱数据
        num_workers=2           # 使用2个子进程加载数据
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # 测试时不需要打乱
        num_workers=2
    )
    
    return train_loader, test_loader

# 2. 定义神经网络模型 ===========================================
class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络模型
    结构：卷积层 -> 池化层 -> 卷积层 -> 池化层 -> 全连接层
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 第一个卷积层：输入通道1(灰度图)，输出通道32，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # 第二个卷积层：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 最大池化层：2x2窗口
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout层：防止过拟合，训练时随机丢弃50%的神经元
        self.dropout = nn.Dropout2d(0.5)
        
        # 全连接层
        # 经过两次池化后，图像尺寸从28x28变为7x7 (28->14->7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64*7*7 -> 128
        self.fc2 = nn.Linear(128, 10)          # 128 -> 10 (10个数字类别)
        
    def forward(self, x):
        """
        前向传播过程
        x: 输入数据 [batch_size, 1, 28, 28]
        """
        # 第一个卷积+激活+池化
        x = self.pool(F.relu(self.conv1(x)))  # [batch_size, 32, 14, 14]
        
        # 第二个卷积+激活+池化
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size, 64, 7, 7]
        
        # Dropout
        x = self.dropout(x)
        
        # 展平操作：将三维特征图展平为一维向量
        x = x.view(-1, 64 * 7 * 7)  # [batch_size, 64*7*7]
        
        # 全连接层
        x = F.relu(self.fc1(x))     # [batch_size, 128]
        x = self.fc2(x)             # [batch_size, 10]
        
        return x

# 3. 训练函数 ===================================================
def train(model, device, train_loader, optimizer, epoch):
    """
    训练模型一个epoch
    """
    model.train()  # 设置为训练模式
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到设备（GPU/CPU）
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = F.cross_entropy(output, target)
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        # 统计信息
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 每100个batch打印一次进度
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # 计算平均损失和准确率
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 4. 测试函数 ===================================================
def test(model, device, test_loader):
    """
    测试模型性能
    """
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    
    # 在测试时不计算梯度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 累加损失
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            
            # 获取预测结果
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
    
    # 计算平均损失和准确率
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

# 5. 可视化函数 =================================================
def visualize_predictions(model, device, test_loader, num_images=10):
    """
    可视化模型预测结果
    """
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 预测
    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    # 转换回CPU用于显示
    images = images.cpu().numpy()
    
    # 显示图像和预测结果
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        axes[i].imshow(images[i][0], cmap='gray')
        axes[i].set_title(f'True: {labels[i].item()}, Pred: {predicted[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 6. 主函数 =====================================================
def main():
    # 设置超参数
    batch_size = 64
    learning_rate = 0.001
    epochs = 5  # 训练轮数，可以增加到10-20获得更好效果
    
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    train_loader, test_loader = load_data(batch_size)
    
    # 创建模型
    print("Creating model...")
    model = SimpleCNN().to(device)
    
    # 打印模型结构
    print("\nModel architecture:")
    print(model)
    
    # 定义优化器（Adam优化器）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 可选：学习率调度器（每3个epoch学习率减半）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # 用于记录训练过程
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # 训练循环
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 测试
        test_loss, test_acc = test(model, device, test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 更新学习率
        scheduler.step()
    
    # 可视化训练过程
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, 'b-', label='Training Accuracy')
    plt.plot(range(1, epochs + 1), test_accs, 'r-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 可视化一些预测结果
    print("Visualizing predictions...")
    visualize_predictions(model, device, test_loader)
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Model saved as 'mnist_cnn.pth'")
    
    # 保存整个模型（包含结构）
    torch.save(model, 'mnist_cnn_full.pth')
    print("Full model saved as 'mnist_cnn_full.pth'")

# 7. 模型使用示例 ===============================================
def load_and_predict():
    """
    加载保存的模型进行预测
    """
    # 加载模型
    device = torch.device("cpu")  # 在CPU上推理
    model = SimpleCNN()
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
    model.eval()
    
    # 创建随机测试数据
    # 在实际应用中，这里应该加载真实的图像数据
    test_image = torch.randn(1, 1, 28, 28)  # 随机生成一个图像
    
    # 预测
    with torch.no_grad():
        output = model(test_image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    return predicted_class, confidence

# 运行主函数
if __name__ == '__main__':
    # 首次运行时训练模型
    main()
    
    # 如果需要加载已有模型进行预测，取消下面的注释
    # load_and_predict()