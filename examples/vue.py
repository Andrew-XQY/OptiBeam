


""" plot 3D PCA with motion"""
# if __name__ == "__main__":
#     image_paths = get_all_file_paths("C:\\Users\\qiyuanxu\\Documents\\DataWarehouse\\MMF\\procIMGs\\processed\\")
#     images = load_images(image_paths, funcs=[np.array, rgb_to_grayscale, split_image, lambda x: x[0].flatten()])
#     pca = visualPCA(n_components=3)
#     pca.fit(images)
#     pca.create_gif("C:\\Users\\qiyuanxu\\Documents\\Results\\beam.gif", fps=30, reverse=True)