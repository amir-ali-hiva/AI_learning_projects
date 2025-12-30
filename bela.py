import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. خواندن تصویر
# -----------------------------
image_path = "input.tif"
image = cv2.imread(image_path)

if image is None:
    raise RuntimeError("Image not found")

# -----------------------------
# 2. grayscale
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3. blur
# -----------------------------
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# -----------------------------
# 4. threshold (مهم!)
# -----------------------------
_, binary = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# -----------------------------
# 5. morphology
# -----------------------------
kernel = np.ones((3, 3), np.uint8)

binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

# -----------------------------
# 6. حذف اجسام خیلی کوچک
# -----------------------------
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

min_area = 150
clean = np.zeros(binary.shape, dtype=np.uint8)

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > min_area:
        clean[labels == i] = 255

# -----------------------------
# 7. شمارش نهایی
# -----------------------------
count, _ = cv2.connectedComponents(clean)
pepper_count = count - 1

print("Pepper count:", pepper_count)

# -----------------------------
# 8. نمایش برای چک کردن
# -----------------------------
plt.figure(figsize=(6,6))
plt.imshow(clean, cmap='gray')
plt.title(f"Pepper count = {pepper_count}")
plt.axis('off')
plt.show()
