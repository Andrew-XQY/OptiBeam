from conftest import *
import cv2

# Example usage
dataset_path = '../../ResultsCenter/dataset/2024-06-06/'
dirs = utils.get_all_file_paths(dataset_path)

crop_areas = processing.select_crop_areas_center(cv2.imread(dirs[-1]), num=3, scale_factor=0.3) # This would mean a total of 4 clicks (2 rectangles)
print("Defined crop areas:", crop_areas)

temp = []
for i in dirs:
    img = cv2.imread(i)
    temp.append(processing.crop_images_from_clicks(crop_areas, img))
    
    
for i, img in enumerate(temp):
    for j, cropped in enumerate(img):
        cv2.imwrite(dataset_path + 'processed/' + f"{i}_{j}.png", cropped)

# visualization.plot_narray(cv2.imread(dirs[0]))  


