""" plot 3D PCA with camera perspective change motion"""
if __name__ == "__main__":
    image_paths = get_all_file_paths("C:\\Users\\qiyuanxu\\Documents\\DataWarehouse\\MMF\\procIMGs\\processed\\")
    images = load_images(image_paths, funcs=[np.array, rgb_to_grayscale, split_image, lambda x: x[0].flatten()])
    pca = visualPCA(n_components=3)
    pca.fit(images)
    pca.create_gif("C:\\Users\\qiyuanxu\\Documents\\Results\\beam.gif", fps=30, reverse=True)


"""multiprocess example"""
# Example 1:
def func1(x):
    return x * 2

def func2(x):
    return x - 1

total_func = utils.combine_functions([func1, func2])
test = [i for i in range(10000)]
utils.apply_multiprocess(total_func)(test)


# Example 2:
@utils.apply_multiprocess
def func(x):
    return x * 2

test = [i for i in range(10000)]
func(test);
