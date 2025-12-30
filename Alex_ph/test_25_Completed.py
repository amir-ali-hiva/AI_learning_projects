import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Load
# ===============================
img = cv2.imread("D:/flash/programing/Python/pythonProject12/AI_learning_projects/Alex_ph/input.tif")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===============================
# 2. Preprocess
# ===============================
gray = cv2.GaussianBlur(gray, (15,9), 0)

_, binary = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = np.ones((3,3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 1)

# ===============================
# 3. Distance peaks (for center)
# ===============================
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
local_max = cv2.dilate(dist_norm, peak_kernel)

peaks = (dist_norm >= local_max - 1e-6) & (dist_norm > 0.3)
peaks = peaks & (binary > 0)
peaks = np.uint8(peaks * 255)

# ===============================
# 4. Contours (for edges)
# ===============================
contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

centers = []

# ---- center peppers (from peaks)
# ---- center peppers (from peak regions, NOT pixels)
num_p, peak_labels = cv2.connectedComponents(peaks)

for label in range(1, num_p):
    mask = np.uint8(peak_labels == label)
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))


# ---- edge peppers (from contours touching border)
h, w = binary.shape
for cnt in contours:
    x,y,ww,hh = cv2.boundingRect(cnt)

    # touching image border?
    if x <= 1 or y <= 1 or x+ww >= w-1 or y+hh >= h-1:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            centers.append((cx, cy))

# ===============================
# 5. Remove duplicates (merge centers)
# ===============================
final_centers = []
min_dist = 19


for c in centers:
    if all(np.hypot(c[0]-fc[0], c[1]-fc[1]) > min_dist for fc in final_centers):
        final_centers.append(c)

# ===============================
# 6. Draw & count
# ===============================
out = img.copy()
for i,(x,y) in enumerate(final_centers):
    cv2.circle(out, (x,y), 6, (0,255,0), 2)
    cv2.putText(out, str(i+1), (x-5,y+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

print("Pepper count:", len(final_centers))

plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
