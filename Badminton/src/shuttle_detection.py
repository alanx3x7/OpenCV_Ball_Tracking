# Imports required
import cv2
import sys
import numpy as np
from scipy import ndimage


def find_white_regions(gray_image):
    """ Finds the white regions of the image, as the shuttle is a distinct white colour
    :param gray_image: The grayscale image to find the shuttle in
    :return thresh_image: The thresholded image with regions that are white
    """

    # First find a mask for regions that have brightness above 225
    white_pixels = gray_image > 225
    thresh_image = gray_image.copy()

    # Then do binary erosion (to rid noise), then dilation (to get original shape)
    struct1 = np.ones((11, 11), dtype=bool)
    white_pixels = ndimage.binary_erosion(white_pixels).astype(white_pixels.dtype)
    white_pixels = ndimage.binary_dilation(white_pixels, structure=struct1).astype(white_pixels.dtype)

    # Use mask to turn pixels above threshold to 255, or 0 otherwise
    black_pixels = np.invert(white_pixels)
    thresh_image[white_pixels] = 255
    thresh_image[black_pixels] = 0

    return thresh_image


def find_connected_components(thresh_image):
    """ Finds the connected components of the thresholded image and filters them based on their shape and size
    :param thresh_image: The thresholded image
    :return labels: A 2d matrix of labels for connected components
    """

    rows, cols = thresh_image.shape

    # First find the connected components of the image
    # num_labels: the number of connected components found in the image
    # labels: a matrix with the labels for each pixel
    # stats: [top_left_x_coord, top_left_y_coord, width, height, area] statistics for each component
    # centroids: [x_coord, y_coord] of the centroid of each component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_image)

    # Do some filtering based on the characteristics of a shuttle
    for i in range(num_labels):

        # Filters out shapes that are too long
        size_ratio = stats[i, 2] / stats[i, 3]
        if size_ratio < 0.5 or size_ratio > 2:
            labels[labels == i] = 0

        # Filters out shapes that fit the bounding box too well (likely noise) or too bad (sparse structure)
        area_ratio = stats[i, 4] / (stats[i, 2] * stats[i, 3])
        if area_ratio < 0.4 or area_ratio > 0.9:
            labels[labels == i] = 0

        # Filters out shapes too small (in proportion to image size)
        if stats[i, 2] < (rows / 60) or stats[i, 3] < (cols / 60):
            labels[labels == i] = 0

    return labels


def colour_connected_components(labels):
    """ Uses the list of labels to colour each label a different colour
    :param labels: 2d matrix of labels of each pixel
    :return labeled_img: 2d matrix of coloured labels
    """

    # Finds the HSV values based on the label number
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Convert to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Set the background labels to black
    labeled_img[label_hue == 0] = 0

    return labeled_img


def main(argv):

    # Checks input argument
    if len(argv) != 1 or int(argv[0]) < 1 or int(argv[0]) > 5:
        print("Please enter a valid image number (1, 2, 3, 4, or 5)!")
        return 1

    img_path = '../data/shuttle_detection_image_' + argv[0] + '.jpg'

    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Thresholding, finding connected components, and filter and label image
    thresh_image = find_white_regions(gray_image)
    labels = find_connected_components(thresh_image)
    labeled_img = colour_connected_components(labels)

    # Overlays the components on original image and saving it
    mask = labels.astype(bool)
    image_bgr[mask] = [255, 0, 255]

    cv2.imwrite("../output/Overlayed_image_" + argv[0] + ".jpg", image_bgr)
    cv2.imwrite("../output/Components_image_" + argv[0] + ".jpg", labeled_img)


if __name__ == "__main__":
    main(sys.argv[1:])
