import glob
import cv2
import os

# this is code to make a video from images in a folder
# I used it to make the training timelapses
# this code was mostly copied from the internet

# code for when image names need to be padded with zeros to be in alphabetical and numerical order 
#for i in range(1, 6000):
#  os.rename('MLstuff/training_images3/image_at_epoch_{}.png'.format(i), 'MLstuff/training_images3/image_at_epoch_{}.png'.format(str(i).zfill(4)))

image_folder = 'interpolation'
video_name = 'interpolation4.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 60, frameSize=(width,height))

for image in images:
  video.write(cv2.imread(os.path.join(image_folder,image)))

cv2.destroyAllWindows()
video.release()