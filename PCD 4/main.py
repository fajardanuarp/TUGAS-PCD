import glob
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def to_gray(img):
    """Ubah citra RGB ke grayscale (uint8)."""
    if img.ndim == 3:
        img = img[..., :3]
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img.astype(np.uint8)


def equalize_histogram(img):
    """Ekualisasi histogram: histogram -> probabilitas -> CDF -> pemetaan level."""
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = np.cumsum(hist / img.size)
    cdf_min = cdf[cdf > 0].min()
    lut = np.round((cdf - cdf_min) / max(1 - cdf_min, 1e-12) * 255)
    return lut.clip(0, 255).astype(np.uint8)[img]


def sharpen(img, sigma=2.0, amount=1.0):
    """Penajaman unsharp mask memakai scipy.ndimage."""
    blur = ndimage.gaussian_filter(img.astype(float), sigma=sigma)
    return np.clip(img + amount * (img - blur), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    # Ambil file .jpg pertama di folder ini (ganti dengan nama file jika perlu)
    path = glob.glob("*.jpg")[0]
    asli = to_gray(iio.imread(path))
    equalized = equalize_histogram(asli)
    sharpened = sharpen(ndimage.median_filter(equalized, size=3))

    hasil = [("Asli", asli), ("Equalized", equalized), ("Equalized + Sharpen", sharpened)]

    for nama, img in hasil:
        print(f"{nama:<20} min={img.min():3d} max={img.max():3d} "
              f"mean={img.mean():6.2f} std={img.std():6.2f}")

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for i, (nama, img) in enumerate(hasil):
        ax[0, i].imshow(img, cmap="gray", vmin=0, vmax=255)
        ax[0, i].set_title(nama)
        ax[0, i].axis("off")
        ax[1, i].hist(img.ravel(), bins=256, range=(0, 255), color="gray")
        ax[1, i].set_title(f"Histogram {nama}")
    plt.tight_layout()
    plt.savefig("hasil.png", dpi=150)
    print("Tersimpan: hasil.png")
    plt.show()