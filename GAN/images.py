# all this was yoinked from GeeksForGeeks

from pathlib import Path
import cv2
import os


def extract_images(video_path):
    # Read the video from specified path
    cam = cv2.VideoCapture(
        str(video_path))

    try:

        # creating a folder named data
        if not os.path.exists('data/images'):
            os.makedirs('data/images')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    while(True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './data/images/' + \
                str(video_path.name).split('.')[
                    0] + '_' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


for path in Path('data/new_vids').rglob('*.MOV'):
    print(path)
    extract_images(path)

for path in Path('data/new_vids').rglob('*.mp4'):
    print(path)
    extract_images(path)
