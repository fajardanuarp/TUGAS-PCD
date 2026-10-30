import cv2
import matplotlib.pyplot as plt

img = cv2.imread("meme.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rotasi(citra, sudut):
    h, w = citra.shape[:2]
    pusat = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(pusat, sudut, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    w_baru = int(h * sin + w * cos)
    h_baru = int(h * cos + w * sin)

    M[0, 2] += (w_baru / 2) - pusat[0]
    M[1, 2] += (h_baru / 2) - pusat[1]

    return cv2.warpAffine(citra, M, (w_baru, h_baru))


rot_45 = rotasi(img, 45)
rot_90 = rotasi(img, 90)
rot_180 = rotasi(img, 180)

daftar = [
    (img, "Citra Asli"),
    (rot_45, "Rotasi 45°"),
    (rot_90, "Rotasi 90°"),
    (rot_180, "Rotasi 180°"),
]

plt.figure(figsize=(14, 8))

for i, (gambar, judul) in enumerate(daftar, start=1):
    plt.subplot(2, 2, i)
    plt.imshow(gambar)
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.show()