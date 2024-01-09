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


from matplotlib import pyplot as plt
from scipy.signal import wiener
import numpy as np

# import numpy as np

# def tiepu(W):
#     Ny, Nx = W.shape
#     W = np.concatenate((W, np.fliplr(W)), axis=1)
#     W = np.concatenate((W, np.flipud(W)), axis=0)
#     fsqr = np.tile(0.5/Nx*(-Nx:Nx-1), (2*Ny, 1))**2 + np.tile(0.5/Ny*(-Ny:Ny-1), (1, 2*Nx))**2
#     W = np.fft.ifft2(np.fft.ifftshift(nan2zero(1./fsqr) * np.fft.fftshift(np.fft.fft2(np.imag(np.exp(-1j*W) * np.fft.ifft2(np.fft.ifftshift(fsqr * np.fft.fftshift(np.fft.fft2(np.exp(1j*W))))))))))
#     U = W[0:Ny, 0:Nx]
#     return U


def nan2zero(M):
    M[~np.isfinite(M)] = 0
    return M

# import numpy as np

# def tiepu(W):
#     Ny, Nx = W.shape
#     W = np.concatenate((W, np.fliplr(W)), axis=1)
#     W = np.concatenate((W, np.flipud(W)), axis=0)
#     X, Y = np.meshgrid(np.arange(-Nx, Nx), np.arange(-Ny, Ny))
#     fsqr = (0.5/Nx*X)**2 + (0.5/Ny*Y)**2
#     fsqr[0, 0] = 1 # avoid division by zero
#     W = np.fft.ifft2(np.fft.ifftshift(np.nan_to_num(1/(fsqr+1e-11)) * 
#                                       np.fft.fftshift(np.fft.fft2(np.imag(np.exp(-1j*W) * np.fft.ifft2(np.fft.ifftshift(fsqr*np.fft.fftshift(np.fft.fft2(np.exp(1j*W))))))))))
#     U = W[:Ny, :Nx].real
#     return U

# import numpy as np

# def tiepu(W):
#     Ny, Nx = W.shape
#     W = np.concatenate((W, np.fliplr(W)), axis=1)
#     W = np.concatenate((W, np.flipud(W)), axis=0)
#     # fsqr = np.tile(0.5/Nx*(-Nx:Nx-1), (2*Ny, 1))
#     # **2 + np.tile(0.5/Ny*(-Ny:Ny-1), (1, 2*Nx))**2
#     W = np.fft.ifft2(np.fft.ifftshift(nan2zero(1./fsqr) * np.fft.fftshift(np.fft.fft2(np.imag(np.exp(-1j*W) * np.fft.ifft2(np.fft.ifftshift(fsqr * np.fft.fftshift(np.fft.fft2(np.exp(1j*W))))))))))
#     U = W[0:Ny, 0:Nx]
#     return U

# def tiepu (W):
#     Ny, Nx = W.shape
#     W = np.concatenate ((W, np.fliplr (W)), axis=1)
#     W = np.concatenate ((W, np.flipud (W)), axis=0)
#     x = np.arange (-Nx, Nx)
#     y = np.arange (-Ny, Ny)
#     X, Y = np.meshgrid (x, y)
#     fsqr = (0.5/Nx * X)**2 + (0.5/Ny * Y)**2
#     W = np.fft.ifft2 (np.fft.ifftshift (nan2zero (1./fsqr) * np.fft.fftshift (np.fft.fft2 (np.imag (np.exp (-1j*W) * np.fft.ifft2 (np.fft.ifftshift (fsqr * np.fft.fftshift (np.fft.fft2 (np.exp (1j*W))))))))))
#     U = W [0:Ny, 0:Nx].real
#     return U

# def tiepu (W):
#     Ny, Nx = W.shape
#     W = np.concatenate ((W, np.fliplr (W)), axis=1)
#     W = np.concatenate ((W, np.flipud (W)), axis=0)
#     x = np.arange (-Nx, Nx)
#     y = np.arange (-Ny, Ny)
#     X, Y = np.meshgrid (x, y)
#     fsqr = (0.5/Nx * X)**2 + (0.5/Ny * Y)**2
#     W = np.fft.ifft2 (np.fft.ifftshift (nan2zero (1./fsqr) * np.fft.fftshift (np.fft.fft2 (np.imag (np.exp (-1j*W) * np.fft.ifft2 (np.fft.ifftshift (fsqr * np.fft.fftshift (np.fft.fft2 (np.exp (1j*W)))))))))).real
#     U = W [0:Ny, 0:Nx]
#     return U

import numpy as np

import numpy as np

def tiepu (W):
    Ny, Nx = W.shape
    W = np.concatenate ((W, np.fliplr (W)), axis=1)
    W = np.concatenate ((W, np.flipud (W)), axis=0)
    fsqr = np.tile (0.5/Nx * np.arange (-Nx, Nx), (2*Ny, 1))**2 + np.tile (0.5/Ny * np.arange (-Ny, Ny).reshape (-1, 1), (1, 2*Nx))**2
    W = np.fft.ifft2 (np.fft.ifftshift (nan2zero (1./fsqr) * np.fft.fftshift (np.fft.fft2 (np.imag (np.exp (-1j*W) * np.fft.ifft2 (np.fft.ifftshift (fsqr * np.fft.fftshift (np.fft.fft2 (np.exp (1j*W))))))))))
    U = W [:Ny, :Nx]
    return U

def peaks(x, y):
    z = 3*(1-x)**2*np.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2)- 1/3*np.exp(-(x+1)**2 - y**2)
    z = 1e-4*z
    return z


def fil(I1):
    Nx,Ny = I1.shape
    Iout = np.tile(I1, (3,3))
    Iout = wiener(Iout, (7, 7))    
    return Iout[Nx:2*Nx,Ny:2*Ny]


def iterative_tiepu(W, iter_n=50):
    phi1 = tiepu(W)
    # phi1 = fil(phi1)
    for k in range(iter_n):
        
        # phi1 = fil(phi1) 
        rsd = np.angle(np.exp(1j*(W-phi1)))
        phi_c = tiepu(rsd)
        # phi_c = fil(phi_c)
        phi1 = phi1+phi_c
        # phi1 = fil(phi1) 
    return phi1

# def iterative_tiepu(W, iter_n=50):
#     # W1 =  fil(W) 
#     phi1 = tiepu(W)
#     # phi1 = fil(phi1)
#     # phi1 = fil(phi1) 
#     for k in range(iter_n):
        
#         # phi1 = fil(phi1) 
#         rsd = np.angle(np.exp(1j*(W-phi1)))
#         # plt.figure(dpi=300)
#         # plt.imshow(rsd)
#         # plt.colorbar()
#         phi_c = tiepu(rsd)
#         # phi_c = fil(phi_c)
#         phi1 = phi1+phi_c
#     # phi1 = fil(phi1) 
#     return phi1


# def iterative_tiepu3(W, iter_n=20):
#     # W1 =  fil(W) 
#     phi1 = tiepu(W)
#     # phi1 = fil(phi1)
#     # phi1 = fil(phi1) 
#     for k in range(iter_n):
        
#         # phi1 = fil(phi1) 
#         # W1 = np.angle(np.exp(1j*phi1))
#         # W = (W+W1)/2
#         # plt.imshow(rsd)
#         # plt.colorbar()
#         # phi_c = tiepu(rsd)
#         # phi_c = fil(phi_c)
#         phi1 = tiepu(W)
#     # phi1 = fil(phi1) 
#     return phi1




if __name__ == '__main__':
    
    pi = np.pi

    pixsize = 5e-3    # pixe size
    nlevel = 0
    m = 256
  
    L = 1.28
    x = np.arange(-L/2, L/2, pixsize)
    xx, yy = np.meshgrid(x,x)
    
    tilt_component = 10*xx
    tilt_component1 = 10*yy
    phi = peaks(xx*5, yy*5)  

    phi = (phi-phi.min())/(phi.max()-phi.min()) * pi* 40
    phi_w = (phi+np.random.normal(0, nlevel, [m,m])) % (2*pi)

    u1 = np.exp(1j*phi_w)
    I0 = np.abs(u1)**2
    # u0 = zeropad(u1, nullpixels)

    # phi_hat = tiepu(phi_w)
    phi_hat = iterative_tiepu(phi_w, 10)
    
    noise = np.random.normal(0, 0.1, [m,m])
    # phi_hat1 = iterative_tiepu(phi_w+tilt_component, 3)
    # # phi_hat = phi_hat1 - tilt_component
    # phi_hat2 = iterative_tiepu(phi_w-tilt_component, 3)
    # phi_hat3 = iterative_tiepu(phi_w+tilt_component1, 3)
    # phi_hat4 = iterative_tiepu(phi_w-tilt_component1, 3)
    
    # phi_hat = (phi_hat1+phi_hat2+phi_hat3 +phi_hat4)/4
    
    # phi_hat = tie_pua2(phi_w, pixsize)
    phi = phi - phi.mean()
    phi_hat = phi_hat - phi_hat.mean()
  
    # phi_hat = torch.from_numpy(cv2.blur(phi_hat.numpy(), (3, 3)))
    # phi = imgcrop(phi, 10)
    plt.figure(dpi=300)
    plt.imshow(phi, cmap='jet')
    plt.title('ground true')
    plt.colorbar()
  
    plt.figure(dpi=300)
    plt.imshow(phi_hat, cmap='jet')
    plt.title('unwarpped phase TIE')
    plt.colorbar()
    
    plt.figure(dpi=300)
    plt.imshow(phi_w, cmap='jet')
    plt.title('warpped phase TIE')
    plt.colorbar()
  
    err = phi_hat - phi
    plt.figure(dpi=300)
    plt.imshow(phi_hat - phi, cmap='jet')
    plt.title('residual')
    plt.colorbar()

    rmse = np.sqrt((err**2).mean())
    print('rmse', rmse)
    
    
