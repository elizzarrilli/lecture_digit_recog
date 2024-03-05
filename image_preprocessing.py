import pandas
import imageio
import glob

final_dataset = pandas.DataFrame()

# loop through each number folder
for i in range(10):
  print(i)
  result = []
  for file_path in glob.glob("dataset_raw/" + str(i) + "/*.jpg"):
    imimage = imageio.v2.imread(file_path)
    imimage = imimage.flatten()
    result.append(imimage)
    
  result_dataframe = pandas.DataFrame(result)
  result_dataframe = result_dataframe.add_prefix('pixel_')
  result_dataframe['digit'] = i
  final_dataset = pandas.concat([result_dataframe, final_dataset], axis=0)
  
final_dataset.to_csv("dataset.csv", index=False)





