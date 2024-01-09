import numpy as np
import matplotlib.pyplot as plt



def iterative_tiepu(W, iter_n=3):
    phi1 = tiepu(W)
    k1 = np.round((phi1-W)/2/np.pi)
    phi2 = k1*2*np.pi + W
    
    for k in range(iter_n):
        err = phi2-phi1
        phic = tiepu(err)
        phi1 = phi1 + phic
        k2 = np.round((phi1-W)/2/np.pi)
        phi2 = k2*2*np.pi + W
        k1 = k2        
    return phi1

# def tiepu (phi_W):
#     Ny, Nx = phi_W.shape
#     phi_W = np.hstack ((phi_W, np.fliplr (phi_W)))
#     phi_W = np.vstack ((phi_W, np.flipud (phi_W)))

    
#     # f = np.sqrt (np.fft.fftfreq (Ny, 1/Ny)**2 + np.fft.fftfreq (Nx, 1/Nx)**2)
#     # f = np.fft.fftshift (f) + 1e-10
#     f = np.tile (0.5/Nx*np.arange (-Nx,Nx).reshape (1,-1), (2*Ny,1))**2 + np.tile (0.5/Ny*np.arange (-Ny,Ny).reshape (-1,1), (1,2*Nx))**2 + 1e-12
#     # f = f+
#     # print(f.shape)
#     # f [f==0] = np.finfo (float).eps # avoid division by zero
#     FT = np.fft.fft2 (np.exp (1j*phi_W))
#     Gamma = np.fft.ifft2 (f**2 * FT)
#     result = np.fft.ifft2 (1/f**2 * np.fft.fft2 (np.imag (np.exp (-1j*phi_W) * Gamma)))

#     return np.real (result[:Ny,:Nx])


# import numpy as np
# from scipy import fftpack

# def tiepu (phi_W):

#     # 计算空间频率

#     phi_W = np.hstack ((phi_W, np.fliplr (phi_W)))
#     phi_W = np.vstack ((phi_W, np.flipud (phi_W)))
#     Ny, Nx = phi_W.shape
#     fx = fftpack.fftfreq (Nx)
#     fy = fftpack.fftfreq (Ny)
#     f = np.sqrt (fx**2 + fy**2)

#     # 计算傅立叶变换和逆傅立叶变换
#     FT = np.fft.fft2
#     IFT = np.fft.ifft2

#     # 计算公式中的各个部分
#     part1 = 1 /(f**2 + 1e-10)
#     part2 = np.exp (1j * phi_W)
#     part3 = IFT (part1 * FT (part2))
#     part4 = np.imag (np.exp (-1j * phi_W) * part3)

#     # 计算最终结果
#     phi_UW = IFT (part1 * FT (part4))

#     return phi_UW.real[:Nx//2,:Nx//2]

# 定义相位解缠算法




def tiepu (W):
    Ny, Nx = W.shape
    W = np.hstack ((W, np.fliplr (W)))
    W = np.vstack ((W, np.flipud (W)))
    fsqr = np.tile (0.5/Nx*np.arange (-Nx,Nx).reshape (1,-1), (2*Ny,1))**2 + np.tile (0.5/Ny*np.arange (-Ny,Ny).reshape (-1,1), (1,2*Nx))**2 + 1e-12
    W = np.fft.ifft2(np.fft.ifftshift (np.nan_to_num (1./fsqr)*np.fft.fftshift (np.fft.fft2 (np.imag (np.exp (-1j*W)*np.fft.ifft2 (np.fft.ifftshift (fsqr*np.fft.fftshift (np.fft.fft2 (np.exp (1j*W))))))))))
    U = W [:Ny,:Nx].real
    return U


# def tiepu (W):
#     Ny, Nx = W.shape
#     W = np.concatenate ((W, np.fliplr (W)), axis=1)
#     W = np.concatenate ((W, np.flipud (W)), axis=0)
#     x = np.arange (-Nx, Nx)
#     y = np.arange (-Ny, Ny)
#     X, Y = np.meshgrid (x, y)
#     fsqr = (0.5/Nx * X)**2 + (0.5/Ny * Y)**2 +1e-11
#     W = np.fft.ifft2 (np.fft.ifftshift (nan2zero (1./fsqr) * np.fft.fftshift (np.fft.fft2 (np.imag (np.exp (-1j*W) * np.fft.ifft2 (np.fft.ifftshift (fsqr * np.fft.fftshift (np.fft.fft2 (np.exp (1j*W)))))))))).real
#     U = W [0:Ny, 0:Nx]
#     return U

# 定义 nan2zero 函数
def nan2zero (x):
    x [np.isnan (x)] = 0
    return x

# 定义 peaks 函数
def peaks (x, y):
    z = 3*(1-x)**2*np.exp (-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp (-x**2-y**2)- 1/3*np.exp (-(x+1)**2 - y**2)
    z = 1e-4*z
    return z

# 设置参数
nlevel = 0.0
pi = np.pi
pixsize = 5e-3 # pixe size
wavlen = 0.5e-3 # wavelength
k = 2*np.pi/wavlen
distance0 = -5
distance = 0.01 # propagation distance
m = 256
kernelsize = distance*wavlen/pixsize/2
nullpixels = int (kernelsize /pixsize)
L = m*pixsize
x = np.arange (-L/2, L/2, pixsize)
yy, xx= np.meshgrid (x,x)

# 生成相位真值
phi = peaks (xx*5, yy*5)
phi = (phi-phi.min ())/(phi.max ()-phi.min ()) * pi* 40

# 添加噪声并将相位折叠至 0-2pi 范围
phi_w = (phi + np.random.normal (0, nlevel, [m,m])) % (2*pi)

# 执行相位解缠算法
unwrapped_phase = tiepu (phi_w)

# 减去分量
unwrapped_phase = unwrapped_phase - unwrapped_phase.mean()
phi = phi - phi.mean()

# 计算与真值的均方误差
mse = np.mean ((unwrapped_phase - phi)**2)

print ("均方误差：", mse)

# 绘制图像和残差图
fig, axs = plt.subplots (1, 3, figsize=(12, 4))

# 绘制原始相位图像
im0 = axs [0].imshow (phi_w, cmap='hsv')
axs [0].set_title ('Wrapped Phase')
fig.colorbar (im0, ax=axs [0])

# 绘制解缠后的相位图像
im1 = axs [1].imshow (unwrapped_phase, cmap='hsv')
axs [1].set_title ('Unwrapped Phase')
fig.colorbar (im1, ax=axs [1])

# 绘制残差图像
im2 = axs [2].imshow (residual, cmap='hsv')
axs [2].set_title ('Residual')
fig.colorbar (im2, ax=axs [2])

plt.tight_layout ()
plt.show ()