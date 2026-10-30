import cv2
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread("meme.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Scaling faktor 0,5 (diperkecil) dan 2 (diperbesar)
scale_05 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
scale_2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

print("Ukuran gambar asli  :", img.shape)
print("Ukuran scaling 0,5  :", scale_05.shape)
print("Ukuran scaling 2    :", scale_2.shape)

# Menampilkan hasil
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title(f"Asli: {img.shape[1]} x {img.shape[0]}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(scale_05)
plt.title(f"Scaling 0,5: {scale_05.shape[1]} x {scale_05.shape[0]}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(scale_2)
plt.title(f"Scaling 2: {scale_2.shape[1]} x {scale_2.shape[0]}")
plt.axis("off")

plt.tight_layout()
plt.show()