# Todas las funciones de la clase

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def leer_imagen(ruta_img):
    img = mpimg.imread(ruta_img)
    return img


def RGB2GRAY(img):
    # 0.299R + 0.587G + 0.114B
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return gray

def RGB2CMYK(img):
    # normalizar
    if img.max() > 1.0:
        img = img / 255.0
    
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # negro
    K = 1 - np.maximum.reduce([R, G, B])

    # fórmulas C, M, Y cuando K < 1
    C = (1 - R - K) / (1 - K)
    M = (1 - G - K) / (1 - K)
    Y = (1 - B - K) / (1 - K)

    return C, M, Y, K


def img_cuantizada(img, L):
    x = img.astype(np.float32)

    x = x / 255.0

    levels = 2**L 
    q = np.round(x * (levels - 1))
    xq = q / (levels - 1)

    return q

def extrae_planos_bits(img):
    if img.ndim == 3 and img.shape[2] == 3:
        img = mpi.RGB2GRAY(img)

    h, w = img.shape
    bit_planes = np.zeros((h, w, 8), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            p = int(img[i, j])
            bin_str = f"{p:08b}"
            for b in range(8):
                bit_planes[i, j, b] = int(bin_str[b]) 

    return bit_planes

def mostrar_planos_bits(bit_planes):
    plt.figure(figsize=(12, 6))
    for b in range(8):
        plt.subplot(2, 4, b+1)
        plt.imshow(bit_planes[..., b], cmap="gray")
        plt.title(f"Plano bit {7-b}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    import numpy as np

def RGB2Lab(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    mask = img > 0.04045
    img_lin = np.where(mask, ((img + 0.055)/1.055)**2.4, img/12.92)

    R, G, B = img_lin[...,0], img_lin[...,1], img_lin[...,2]

    X = 0.4124564*R + 0.3575761*G + 0.1804375*B
    Y = 0.2126729*R + 0.7151522*G + 0.0721750*B
    Z = 0.0193339*R + 0.1191920*G + 0.9503041*B

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X/Xn, Y/Yn, Z/Zn

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16/116)

    fx, fy, fz = f(x), f(y), f(z)

    L = (116*fy - 16)
    a = 500*(fx - fy)
    b = 200*(fy - fz)

    Lab = np.stack([L, a, b], axis=-1)
    return Lab

def histograma(img, L):
    h = np.empty((2**L))

    for i in range(2**L):
        h[i] = np.sum(img == i)

    return h.astype(int)

def Img_Histograma(img):
    img = np.array(img).clip(0,255).astype("uint8")
    h = histograma(img)

    _,axes = plt.subplots(1,2)
    axes = axes.flatten()

    axisImg = axes[0]
    axisH = axes[1]

    axisImg.imshow(img,cmap="gray",vmax = 255)
    axisH.bar(range(256),h,width=1.0)
    axisH.set_xlim(0,255)

    plt.show()

def Img_Histograma_RGB(img):
    img = np.array(img).clip(0, 255).astype("uint8")
    hR = histograma(img[:,:,0])
    hG = histograma(img[:,:,1])
    hB = histograma(img[:,:,2])

    _, axes = plt.subplots(1, 2)
    axes = axes.flatten()

    axisImg = axes[0]
    axisH = axes[1]

    axisImg.imshow(img[..., :3], vmax=255)
    axisH.bar(range(256), hR, color='r', width=1.0, alpha=0.5, label='Red')
    axisH.bar(range(256), hG, color='g', width=1.0, alpha=0.5, label='Green')
    axisH.bar(range(256), hB, color='b', width=1.0, alpha=0.5, label='Blue')
    axisH.set_xlim(0, 255)
    axisH.legend()

    plt.show()

def ecualización_histograma(img):
    h = histograma(img)
    total = h.sum()
    h_norm = h / total

    h_acumulativo = h_norm.cumsum()

    # Matriz final con la transformación. Cada índice corresponde a una
    # intensidad original, y el valor en ese índice es la nueva intensidad.
    # Convertir a uint8 también hace automáticamente la operación de floor.
    eq = (h_acumulativo * 255).astype("uint8")

    # Aplicar la transformación a un pixel individual. A partir de la intensidad
    # original (pixel), obtiene la nueva intensidad (eq[pixel]).
    def ecualización_pixel(pixel):
        return eq[pixel]

    # Vectorizar la función para aplicarla a toda la matriz
    ecualización_img = np.vectorize(ecualización_pixel)
    img_eq = ecualización_img(img)

    return img_eq

def ecualización_histograma_RGB(img):
    img_eq_R = ecualización_histograma(img[:,:,0])
    img_eq_G = ecualización_histograma(img[:,:,1])
    img_eq_B = ecualización_histograma(img[:,:,2])

    (height, width) = img.shape[:2]
    res = np.empty((height, width, 3))
    res[:, :, 0] = img_eq_R
    res[:, :, 1] = img_eq_G
    res[:, :, 2] = img_eq_B

    return res.astype("uint8")


def mse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2, max_pixel=255.0):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    rmse = np.sqrt(mse)
    return 20 * np.log10(max_pixel / rmse)

def histograma_cuan(img, L):
    h = np.zeros(2**L)
    alto, ancho = img.shape 
    for x in range(alto):
        for y in range(ancho):
            rk = int(img[x,y])
            h[rk] += 1
    return h

def histograma_general(img, bins=256):
    min_val = np.floor(img.min())
    max_val = np.ceil(img.max())
    h = np.zeros(bins)
    rango = max_val - min_val
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            idx = int((img[x, y] - min_val) * (bins - 1) / rango)
            h[idx] += 1
    return h

def agregar_ruido_gaussiano(img, media=0, sigma=20):
    ruido = np.random.normal(media, sigma, img.shape)
    img_ruidosa = img + ruido
    return img_ruidosa, ruido


def convolucion2D(img, kernel):
    img = img.astype(np.float32)
    kernel = np.array(kernel, dtype=np.float32)

    kernel = np.flipud(np.fliplr(kernel))

    alto, ancho = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    salida = np.zeros_like(img, dtype=np.float32)

    for i in range(alto):
        for j in range(ancho):
            region = padded[i:i+kh, j:j+kw]
            salida[i, j] = np.sum(region * kernel)

    salida = np.clip(salida, 0, 255)
    return salida.astype(np.uint8)


def agregar_ruido_sal_pimienta(img, prob=0.05):
    salida = np.copy(img)
    alto, ancho = img.shape

    ruido = np.random.rand(alto, ancho)

    salida[ruido < (prob / 2)] = 0  # pimienta
    salida[ruido > 1 - (prob / 2)] = 255  # sal

    return salida


def kernel_gaussiano(size, sigma):
    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  
    return kernel


def mostrar_imagen(img):
    img = img.clip(0,255)
    img = img.astype('uint8')
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.show()


def mostrar_matriz_imagenes(imagenes, titulos):
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(imagenes)
    cols = int(np.ceil(np.sqrt(n)))
    filas = int(np.ceil(n / cols))

    plt.figure(figsize=(4 * cols, 4 * filas))

    for i in range(n):
        plt.subplot(filas, cols, i + 1)
        plt.imshow(imagenes[i], cmap="gray")
        plt.title(titulos[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()



def down(img, K, M):
    img_filt = convolucion2D(img, K)
    img_down = img_filt[::M, ::M]
    return img_filt, img_down
    
def up(img, K, L):
    alto, ancho = img.shape
    up_zeros = np.zeros((alto * L, ancho * L), dtype=np.float32)

    up_zeros[::L, ::L] = img

    up_filt = convolucion2D(up_zeros, K * (L**2))

    return up_zeros, up_filt


def mediana(img, tam_filtro):
    pad = tam_filtro // 2
    padded = np.pad(img, pad, mode='constant')
    img_out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+tam_filtro, j:j+tam_filtro]
            img_out[i, j] = np.median(region)
    return img_out.astype(np.uint8)


def minimo(img, tam_filtro):
    pad = tam_filtro // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    img_out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+tam_filtro, j:j+tam_filtro]
            img_out[i, j] = np.min(region)
    return img_out.astype(np.uint8)


def maximo(img, tam_filtro):
    pad = tam_filtro // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    img_out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+tam_filtro, j:j+tam_filtro]
            img_out[i, j] = np.max(region)
    return img_out.astype(np.uint8)


def moda(img, tam_filtro):
    pad = tam_filtro // 2
    padded = np.pad(img, pad, mode='constant')
    img_out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+tam_filtro, j:j+tam_filtro].flatten()
            vals, counts = np.unique(region, return_counts=True)
            img_out[i, j] = vals[np.argmax(counts)]
    return img_out.astype(np.uint8)


def alfa_recortada(img, tam_fila, alpha=1):
    pad = tam_fila // 2
    padded = np.pad(img, pad, mode='constant')
    img_out = np.zeros_like(img)
    d = tam_fila * tam_fila
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = np.sort(padded[i:i+tam_fila, j:j+tam_fila].flatten())
            if 2 * alpha < d:
                region = region[alpha : d - alpha]
            img_out[i, j] = np.mean(region)
    return img_out.astype(np.uint8)
