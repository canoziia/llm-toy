"""
作业3：使用物理信息神经网络求解泊松方程

问题描述：
在立方体区域 [-1,1]³ 中求解泊松方程
∇²φ = -ρ(x,y,z)
其中 ρ(x,y,z) = 100xyz²
边界条件：φ = 0 on boundary

TODO: 请完成标记为 TODO 的部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ==================== 1. 定义神经网络 ====================
class PINN(nn.Module):
    """
    物理信息神经网络
    TODO: 定义网络层结构
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PINN, self).__init__()
        # TODO: 定义网络层
        # 提示：使用 nn.Linear 定义全连接层
        # 建议 3-5 层，每层 128-512 个神经元
        pass

    def forward(self, x):
        # TODO: 定义前向传播
        # 提示：使用 tanh 或其他激活函数
        pass


# ==================== 2. 采样函数 ====================
def sample_points_in_cube(N, device="cpu"):
    """
    在立方体域 [-1, 1]³ 内随机采样点

    TODO: 实现域内采样
    提示：使用 torch.rand 生成 [0,1] 范围的随机数，然后缩放到 [-1,1]

    返回：
        torch.Tensor: 形状 (N, 3) 的张量
    """
    pass


def sample_points_on_boundary(N, device="cpu"):
    """
    在立方体的 6 个边界面上采样点

    TODO: 实现边界采样
    提示：立方体有 6 个面（x=±1, y=±1, z=±1），在每个面上采样

    返回：
        torch.Tensor: 边界点集合
    """
    pass


# ==================== 3. 物理方程 ====================
def charge_distribution(r):
    """
    定义电荷分布 ρ(x,y,z) = 100xyz²

    参数：
        r: 位置坐标，形状 (N, 3)

    返回：
        电荷密度值
    """
    # TODO: 实现电荷分布函数
    pass


def compute_pde_residual(model, r):
    """
    计算泊松方程残差：∇²φ + ρ = 0

    TODO: 使用自动微分计算拉普拉斯算子

    提示：
    1. 首先计算一阶导数（梯度）
    2. 然后对每个方向计算二阶导数
    3. 求和得到拉普拉斯算子

    参数：
        model: PINN 模型
        r: 位置坐标（需要 requires_grad=True）

    返回：
        PDE 残差
    """
    # TODO: 计算 φ = model(r)

    # TODO: 计算一阶导数 ∇φ
    # 提示：使用 torch.autograd.grad，设置 create_graph=True

    # TODO: 计算二阶导数（拉普拉斯算子）
    # 提示：对 x, y, z 三个方向分别计算二阶导数并求和

    # TODO: 计算电荷密度 ρ

    # TODO: 返回 PDE 残差：∇²φ + ρ
    pass


# ==================== 4. 训练函数 ====================
def train(model, optimizer, num_epochs, device="cpu"):
    """
    训练 PINN 模型

    TODO: 实现训练循环

    参数：
        model: PINN 模型
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备
    """
    losses = []

    # TODO: 采样边界点（可以固定）

    for epoch in range(num_epochs):
        # TODO: 每轮重新采样域内点

        # TODO: 前向传播：计算边界点的 φ 值

        # TODO: 计算边界损失（边界条件 φ = 0）

        # TODO: 计算 PDE 残差和 PDE 损失

        # TODO: 计算总损失（边界损失 + β * PDE 损失）

        # TODO: 反向传播和优化

        # TODO: 记录损失并定期打印

        pass

    return losses


# ==================== 5. 主程序 ====================
if __name__ == "__main__":
    # TODO: 设置超参数
    # input_dim = 3
    # hidden_dim = 256
    # output_dim = 1
    # num_epochs = 10000
    # learning_rate = 0.001

    # TODO: 初始化模型和优化器

    # TODO: 训练模型

    # TODO: 保存模型
    # torch.save(model.state_dict(), 'pinn.pth')

    # TODO: 可视化训练曲线
    # 提示：使用 matplotlib 绘制损失曲线

    # TODO: 测试和可视化结果
    # 提示：在测试点上预测电势，绘制等高线图或切片图

    pass
