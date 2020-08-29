import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def main():

    # Container to store images for video output
    img_array =[]

    # Import the video
    # cap = cv2.VideoCapture("Videos/VID_20191114_221955.mp4")
    cap = cv2.VideoCapture("Videos/VID_20191114_221152.mp4")

    # Read the first frame
    ret, frame = cap.read()

    # Grayscale image (don't know if blur helps rn, should look into it)
    prev_frame = cv2.blur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5))
    # curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Suppress all points that aren't bright enough (i.e. not white enough)
    prev_frame[prev_frame < 0.7 * np.max(prev_frame)] = 0

    count = 0                   # Counter just for printing frame number
    previous_ind = [0, 0]       # Previous max index of brightest point

    # Loops through all of the images
    while ret:

        # Gets the next frame
        ret, frame = cap.read()

        # Stops when no more images
        if not ret:
            break

        # Convert to grayscale and then blur
        # next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.blur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5))

        # Threshold points that aren't white enough
        curr_frame[curr_frame < 0.5 * np.max(curr_frame)] = 0

        # Subtract from the previous image to get the difference
        difference = abs(curr_frame - prev_frame)

        # Threshold so that points that are too different or not different enough are not counted
        difference[difference > 240] = 0  # iirc noise tends to have a huge difference, so gets rid of it
        difference[difference < 0.7 * np.max(difference)] = 0

        # Median filter to get rid of salt and pepper
        difference = cv2.medianBlur(difference, 7)

        # Finds the maximum signal
        ind = np.where(difference == np.max(difference))

        # If there is no maximum in this current photo, takes the coordinate of the maximum of the previous photo
        if ind[1].size > 1000:
            ind[1][0] = previous_ind[1]
            ind[0][0] = previous_ind[0]

        # Draw that point on the image
        frame = cv2.circle(frame, (ind[1][0], ind[0][0]), 10, color=[0, 255, 0], thickness=-1)

        # Updates the location of the maximum
        previous_ind[1] = ind[1][0]
        previous_ind[0] = ind[0][0]

        # Append to the image container for video creation later
        img_array.append(frame)

        # Makes the current frame the previous frame
        prev_frame = curr_frame

        # Update count, print count
        count = count + 1
        print("Frame: %d" % count)

        # Just here because my comp can't handle too much memory - can up it if yours can lol
        if count > 150:
            break

    # Create the video
    height, width = prev_frame.shape
    size = (width, height)
    out = cv2.VideoWriter('cool.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start = time.time()
    main()
    print(time.time() - start)