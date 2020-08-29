import cv2
import os
import time
import numpy as np
import skvideo
skvideo.setFFmpegPath(
    'C:\\Users\\alanx\\Documents\\JHU\\Courses\\EN.601.461_Computer_Vision\\ffmpeg-20191115-bfa8272-win64-shared\\bin')
import skvideo.io
import matplotlib.pyplot  as plt
from scipy import ndimage


def main():
    img_array =[]
    # cap = cv2.VideoCapture("Videos/VID_20191114_221955.mp4")
    cap = cv2.VideoCapture("../data/VID_20191114_221152.mp4")
    ret, frame = cap.read()
    # curr_frame = cv2.blur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5))
    curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr_frame[curr_frame < 0.7 * np.max(curr_frame)] = 0
    count = 0
    previous_ind = [0, 0]

    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # next_frame = cv2.blur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5))
        next_frame[next_frame < 0.5 * np.max(next_frame)] = 0
        edges = abs(next_frame - curr_frame)
        print(edges.shape)
        edges[edges > 240] = 0
        edges[edges < 0.7 * np.max(edges)] = 0
        edges = cv2.medianBlur(edges, 7)
        print(edges.shape)

        ind = np.where(edges == np.max(edges))
        print(ind)

        # struct1 = np.ones((5, 5), dtype=bool)
        # edges = ndimage.binary_dilation(edges, structure=struct1).astype(np.uint8)

        if 1 == 0:
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
            min_size = 10
            max_size = 400
            # your answer image
            sizes = stats[1:, -1]
            nb_components = nb_components - 1
            print(nb_components)
            # for every component in the image, you keep it only if it's above min_size
            for i in range(0, nb_components):
                if sizes[i] < min_size or sizes[i] > max_size:
                    edges[output == i + 1] = 0

        if ind[1].size > 1000:
            ind[1][0] = previous_ind[1]
            ind[0][0] = previous_ind[0]
        frame = cv2.circle(frame, (ind[1][0], ind[0][0]), 10, color=[0, 255, 0], thickness=-1)
        previous_ind[1] = ind[1][0]
        previous_ind[0] = ind[0][0]

        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        img_array.append(frame)
        # plt.imshow(edges)
        # plt.show()
        curr_frame = next_frame
        count = count + 1
        print("Frame: %d" % count)
        if count > 150:
            break
        continue
        edges = cv2.Canny(next_frame, 100, 200)
        plt.imshow(edges)
        plt.show()
        continue

        # Components code
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
        min_size = 5
        max_size = 100
        # your answer image
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        print(nb_components)
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] < min_size:
                edges[output == i + 1] = 0


        print(time.time() - start)

        plt.imshow(frame)
        #plt.imshow(edges)
        #plt.imshow(next_frame)
        plt.show()

    #skvideo.io.vwrite("video.mp4", img_array)
    height, width = curr_frame.shape
    size = (width, height)
    print(curr_frame.shape)
    out = cv2.VideoWriter('cool.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start = time.time()
    main()
    print(time.time() - start)