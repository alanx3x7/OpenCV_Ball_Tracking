import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

img_path1 = 'Photos/4_Set_2_Stationary_Bird_76_.jpg'
# img_path1 = 'Photos/6_Set_2_Stationary_Bird_87.5_.jpg'
img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print(np.max(gray1))

whites = (gray1 > 0.8 * np.max(gray1)).astype(np.uint8)
# struct1 = np.ones((11, 11), dtype=bool)
# whites = ndimage.binary_dilation(whites, structure=struct1).astype(np.uint8)

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(whites, connectivity=8)
print(nb_components)
print(output)
print(stats)
print(centroids)
sizes = stats[1:, -1]
nb_components = nb_components - 1
# min_size = 20
# max_size = 45
for i in range(0, nb_components):
    # if min_size <= sizes[i] <= max_size:
        x = int(centroids[i + 1][0])
        y = int(centroids[i + 1][1])
        img1 = cv2.circle(img1, (x, y), 6, color=[0, 255, 0])

plt.imshow(img1)
plt.show()
exit()


ret, labels = cv2.connectedComponents(whites)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0
    return labeled_img

labeled = imshow_components(labels)
plt.imshow(labeled)
plt.show()
exit()



# print(whites5.shape)
# whites5 = np.invert(whites5)
# img1[whites5] = [0, 0, 0]
# print(img1.shape)
# plt.imshow(np.invert(whites5))
# plt.show()

cv2.imwrite("Output/4_Set_2_Stationary_Bird_76_gray.png", gray1)
