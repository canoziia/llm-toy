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
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # TODO: 定义前向传播
        # 提示：使用 tanh 或其他激活函数
        return self.net(x)


# ==================== 2. 采样函数 ====================
def sample_points_in_cube(N, device="cpu"):
    """
    在立方体域 [-1, 1]³ 内随机采样点

    TODO: 实现域内采样
    提示：使用 torch.rand 生成 [0,1] 范围的随机数，然后缩放到 [-1,1]

    返回：
        torch.Tensor: 形状 (N, 3) 的张量
    """
    # torch.rand(N, 3) in [0, 1], scale to [-1, 1]
    points = 2 * torch.rand(N, 3, device=device) - 1
    return points


def sample_points_on_boundary(N, device="cpu"):
    """
    在立方体的 6 个边界面上采样点

    TODO: 实现边界采样
    提示：立方体有 6 个面（x=±1, y=±1, z=±1），在每个面上采样

    返回：
        torch.Tensor: 边界点集合
    """
    N_per_face = N // 6
    faces = []

    # x = 1 face
    y = 2 * torch.rand(N_per_face, 1, device=device) - 1
    z = 2 * torch.rand(N_per_face, 1, device=device) - 1
    x = torch.ones_like(y)
    faces.append(torch.cat([x, y, z], dim=1))

    # x = -1 face
    y = 2 * torch.rand(N_per_face, 1, device=device) - 1
    z = 2 * torch.rand(N_per_face, 1, device=device) - 1
    x = -torch.ones_like(y)
    faces.append(torch.cat([x, y, z], dim=1))

    # y = 1 face
    x = 2 * torch.rand(N_per_face, 1, device=device) - 1
    z = 2 * torch.rand(N_per_face, 1, device=device) - 1
    y = torch.ones_like(x)
    faces.append(torch.cat([x, y, z], dim=1))

    # y = -1 face
    x = 2 * torch.rand(N_per_face, 1, device=device) - 1
    z = 2 * torch.rand(N_per_face, 1, device=device) - 1
    y = -torch.ones_like(x)
    faces.append(torch.cat([x, y, z], dim=1))

    # z = 1 face
    x = 2 * torch.rand(N_per_face, 1, device=device) - 1
    y = 2 * torch.rand(N_per_face, 1, device=device) - 1
    z = torch.ones_like(x)
    faces.append(torch.cat([x, y, z], dim=1))

    # z = -1 face
    x = 2 * torch.rand(N_per_face, 1, device=device) - 1
    y = 2 * torch.rand(N_per_face, 1, device=device) - 1
    z = -torch.ones_like(x)
    faces.append(torch.cat([x, y, z], dim=1))

    return torch.cat(faces, dim=0)


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
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    return 100.0 * x * y * z**2


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
    phi = model(r)

    # TODO: 计算一阶导数 ∇φ
    # 提示：使用 torch.autograd.grad，设置 create_graph=True
    grad_phi = torch.autograd.grad(
        phi, r, grad_outputs=torch.ones_like(phi), create_graph=True
    )[0]

    # TODO: 计算二阶导数（拉普拉斯算子）
    # 提示：对 x, y, z 三个方向分别计算二阶导数并求和
    laplacian_phi = 0
    for i in range(3):
        grad2_phi_i = torch.autograd.grad(
            grad_phi[:, i],
            r,
            grad_outputs=torch.ones_like(grad_phi[:, i]),
            create_graph=True,
        )[0][:, i]
        laplacian_phi += grad2_phi_i

    # TODO: 计算电荷密度 ρ
    rho = charge_distribution(r)

    # TODO: 返回 PDE 残差：∇²φ + ρ
    return laplacian_phi.unsqueeze(1) + rho.unsqueeze(1)


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

    # 定义采样点数量
    N_boundary = 1200
    N_pde = 10000
    # 定义损失权重
    beta = 1.0

    # TODO: 采样边界点（可以固定）
    r_boundary = sample_points_on_boundary(N_boundary, device)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # TODO: 每轮重新采样域内点
        r_pde = sample_points_in_cube(N_pde, device)
        r_pde.requires_grad = True

        # === 计算边界损失 ===
        # TODO: 前向传播：计算边界点的 φ 值
        phi_boundary = model(r_boundary)
        # TODO: 计算边界损失（边界条件 φ = 0）
        loss_boundary = F.mse_loss(phi_boundary, torch.zeros_like(phi_boundary))

        # === 计算 PDE 损失 ===
        # TODO: 计算 PDE 残差和 PDE 损失
        pde_residual = compute_pde_residual(model, r_pde)
        loss_pde = F.mse_loss(pde_residual, torch.zeros_like(pde_residual))

        # TODO: 计算总损失（边界损失 + β * PDE 损失）
        total_loss = loss_boundary + beta * loss_pde

        # TODO: 反向传播和优化
        total_loss.backward()
        optimizer.step()

        # TODO: 记录损失并定期打印
        losses.append(total_loss.item())
        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.6f}, Boundary Loss: {loss_boundary.item():.6f}, PDE Loss: {loss_pde.item():.6f}"
            )

    return losses


# ==================== 5. 主程序 ====================
if __name__ == "__main__":
    # TODO: 设置超参数
    input_dim = 3
    hidden_dim = 256
    output_dim = 1
    num_epochs = 10000
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: 初始化模型和优化器
    model = PINN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TODO: 训练模型
    losses = train(model, optimizer, num_epochs, device)

    # TODO: 保存模型
    torch.save(model.state_dict(), "pinn.pth")
    print("Model saved to pinn.pth")

    # TODO: 可视化训练曲线
    # 提示：使用 matplotlib 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale("log")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.grid(True)
    plt.savefig("training_curve.png")

    # TODO: 测试和可视化结果
    # 提示：在测试点上预测电势，绘制等高线图或切片图
    model.eval()
    with torch.no_grad():
        # 创建一个 z=0 的平面网格
        grid_res = 100
        x = torch.linspace(-1, 1, grid_res)
        y = torch.linspace(-1, 1, grid_res)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = torch.zeros_like(X)

        grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).to(
            device
        )

        # 预测电势
        phi_pred = model(grid_points).cpu().numpy().reshape(grid_res, grid_res)

        # 绘制电势分布图
        plt.figure(figsize=(8, 6))
        plt.contourf(X.numpy(), Y.numpy(), phi_pred, levels=50, cmap="viridis")
        plt.colorbar(label="Potential φ")
        plt.title("Potential Distribution at z=0")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.savefig("potential_distribution_z0.png")

    # 验证准确性
    print("\nVerifying accuracy on new test points...")
    N_test = 2000
    test_boundary_points = sample_points_on_boundary(N_test, device)
    test_pde_points = sample_points_in_cube(N_test, device)
    test_pde_points.requires_grad = True

    model.eval()
    # 验证边界条件
    phi_boundary_test = model(test_boundary_points)
    boundary_error = torch.mean(phi_boundary_test**2).item()
    print(f"Mean Squared Error on boundary: {boundary_error:.4e}")

    # 验证 PDE 残差
    pde_residual_test = compute_pde_residual(model, test_pde_points)
    pde_error = torch.mean(pde_residual_test**2).item()
    print(f"Mean Squared Error of PDE residual: {pde_error:.4e}")
