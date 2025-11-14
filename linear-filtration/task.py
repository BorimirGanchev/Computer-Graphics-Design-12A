"""
Линейна филтрация на растерни полутонови изображения (grayscale).
Познава: извършва 2D свиване (convolution / correlation) с произволно ядро (kernel),
дава възможност потребителят да въвежда коефициентите на филтъра, scale и offset,
и да избира гранична стратегия (zero, reflect, edge).

Как се използва (пример):
python linear_filter.py --input lena.png --output lena_sharp.png \
    --kernel "0 -1 0; -1 5 -1; 0 -1 0" --scale 1 --offset 0 --mode reflect

Примери за ядра (kernel):
- identity:      0 0 0; 0 1 0; 0 0 0
- box blur 3x3:  1 1 1; 1 1 1; 1 1 1  (scale=9)
- gaussian 3x3:  1 2 1; 2 4 2; 1 2 1  (scale=16)
- sharpen:       0 -1 0; -1 5 -1; 0 -1 0
- emboss:        -2 -1 0; -1 1 1; 0 1 2  (offset=128)

Автор: Автоматично генериран примерен код
"""

from PIL import Image
import numpy as np
import argparse
import sys


def parse_kernel(text: str) -> np.ndarray:
    """Прочита kernel от низ със синтаксис: "a b c; d e f; g h i"
    Точки и запетаи могат да разделят редовете. Връща numpy float64 матрица.
    """
    rows = [r.strip() for r in text.strip().split(';') if r.strip()]
    mat = []
    for r in rows:
        parts = [p for p in r.replace(',', ' ').split() if p]
        mat.append([float(x) for x in parts])
    # проверка за правоъгълна и нечетен размер
    h = len(mat)
    w = len(mat[0])
    for row in mat:
        if len(row) != w:
            raise ValueError('Kernel rows must have same length')
    if h % 2 == 0 or w % 2 == 0:
        raise ValueError('Kernel dimensions should be odd (e.g. 3x3, 5x5)')
    return np.array(mat, dtype=np.float64)


def pad_image(img: np.ndarray, pad_h: int, pad_w: int, mode: str):
    """Pad image according to mode: 'zero', 'reflect', 'edge' (replicate)"""
    if mode == 'zero':
        return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    if mode == 'reflect':
        return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    if mode == 'edge':
        return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    raise ValueError('Unsupported pad mode: ' + mode)


def linear_filter_gray(img: np.ndarray, kernel: np.ndarray, scale: float = None, offset: float = 0.0, mode: str = 'reflect') -> np.ndarray:
    """Apply linear filter (convolution-like) to a 2D grayscale image.

    - img: 2D numpy array (H x W), dtype uint8 or similar
    - kernel: 2D numpy array (kh x kw) with odd sizes
    - scale: optional scalar to divide the convolution sum. If None, defaults to sum(kernel) or 1 if sum=0.
    - offset: value added after scaling (useful for emboss etc.)
    - mode: padding mode for borders: 'zero', 'reflect', 'edge'

    Returns new image as uint8 ndarray.
    """
    if img.ndim != 2:
        raise ValueError('Image must be 2D grayscale')
    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2

    padded = pad_image(img, ph, pw, mode)
    out = np.zeros_like(img, dtype=np.float64)

    # determine scale
    ksum = kernel.sum()
    if scale is None:
        if abs(ksum) < 1e-12:
            scale = 1.0
        else:
            scale = float(ksum)

    # convolution (cross-correlation orientation: no kernel flip) -- most image APIs use correlation,
    # but mathematical convolution flips kernel. We implement correlation to match common filter UIs.
    H, W = img.shape
    for y in range(H):
        for x in range(W):
            region = padded[y:y+kh, x:x+kw]
            value = (region * kernel).sum() / scale + offset
            out[y, x] = value

    # clamp to [0,255]
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def load_image_grayscale(path: str) -> np.ndarray:
    im = Image.open(path).convert('L')
    return np.array(im)


def save_image_gray(arr: np.ndarray, path: str):
    im = Image.fromarray(arr)
    im.save(path)


def build_argparser():
    p = argparse.ArgumentParser(description='Линейна филтрация на полутоново изображение')
    p.add_argument('--input', '-i', required=True, help='Път към входното изображение (grayscale or RGB)')
    p.add_argument('--output', '-o', required=True, help='Файл за резултата')
    p.add_argument('--kernel', '-k', required=True, help='Kernel като текст: "1 0 -1; 2 0 -2; 1 0 -1"')
    p.add_argument('--scale', '-s', type=float, default=None, help='Scale (по подразбиране: sum(kernel) или 1 ако сумата е 0)')
    p.add_argument('--offset', type=float, default=0.0, help='Offset добавен след скалиране (напр. 128 за emboss)')
    p.add_argument('--mode', choices=['zero', 'reflect', 'edge'], default='reflect', help='Гранична стратегия (padding)')
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    img = load_image_grayscale(args.input)
    kernel = parse_kernel(args.kernel)
    out = linear_filter_gray(img, kernel, scale=args.scale, offset=args.offset, mode=args.mode)
    save_image_gray(out, args.output)
    print(f'Готово — записах {args.output}')


if __name__ == '__main__':
    main()
