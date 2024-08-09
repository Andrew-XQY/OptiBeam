from conftest import *
import cv2
from PIL import Image

path = 'C:\\Users\\qiyuanxu\\Documents\\DataWarehouse\\TEST\\1720421563327663700.png'
save_to = 'C:\\Users\\qiyuanxu\\Desktop\\test.png'

img = cv2.imread(path)
crop_areas = processing.select_crop_areas_center(img, num=2, scale_factor=0.5) 
print(crop_areas)


img = processing.crop_image_from_coordinates(img, crop_areas)
cv2.imwrite(save_to, img)
