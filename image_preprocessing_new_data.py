import pandas
import imageio
import glob

results = []
filename_list = []

for file_path in glob.glob("new_data/*.jpg"):
	filename = file_path.split("/")[-1]
	filename_list.append(filename)	
	imimage = imageio.v2.imread(file_path) 
	imimage = imimage.flatten()
	results.append(imimage)

new_data = pandas.DataFrame(results)
new_data = new_data.add_prefix('pixel_')
new_data['filename'] = filename_list

new_data.to_csv("new_data.csv", index = False)






