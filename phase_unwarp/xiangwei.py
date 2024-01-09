import numpy as np
import matplotlib.pyplot as plt

# 定义相位解缠算法
def tiepu (W):
    Ny, Nx = W.shape
    W = np.concatenate ((W, np.fliplr (W)), axis=1)
    W = np.concatenate ((W, np.flipud (W)), axis=0)
    x = np.arange (-Nx, Nx)
    y = np.arange (-Ny, Ny)
    X, Y = np.meshgrid (x, y)
    fsqr = (0.5/Nx * X)**2 + (0.5/Ny * Y)**2
    W = np.fft.ifft2 (np.fft.ifftshift (nan2zero (1./fsqr) * np.fft.fftshift (np.fft.fft2 (np.imag (np.exp (-1j*W) * np.fft.ifft2 (np.fft.ifftshift (fsqr * np.fft.fftshift (np.fft.fft2 (np.exp (1j*W)))))))))).real
    U = W [0:Ny, 0:Nx]
    return U

# 定义 nan2zero 函数
def nan2zero (x):
    x [np.isnan (x)] = 0
    return x

# 定义 peaks 函数
def peaks (x, y):
    z = 3*(1-x)**2*np.exp (-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp (-x**2-y**2)- 1/3*np.exp (-(x+1)**2 - y**2)
    z = 1e-4*z
    return z

# 生成测试相位数据
Ny = 512
Nx = 512
x = np.linspace (-5, 5, Nx)
y = np.linspace (-5, 5, Ny)
X, Y = np.meshgrid (x, y)
phase = peaks (X, Y)*40*3.14

# 将相位折叠至 0-2pi 范围
# phase = np.mod (phase, 2 * np.pi)

# 执行相位解缠算法
unwrapped_phase = tiepu (phase)

# 计算与真值的均方误差
mse = np.mean ((unwrapped_phase - phase)**2)

print ("均方误差：", mse)

# 绘制图像和残差图
fig, axs = plt.subplots (1, 3, figsize=(12, 4))

# 绘制原始相位图像
axs [0].imshow (phase, cmap='hsv')
axs [0].set_title ('Original Phase')

# 绘制解缠后的相位图像
axs [1].imshow (unwrapped_phase, cmap='hsv')
axs [1].set_title ('Unwrapped Phase')

# 绘制残差图像
residual = unwrapped_phase - phase
axs [2].imshow (residual, cmap='hsv')
axs [2].set_title ('Residual')

plt.tight_layout ()
plt.show ()