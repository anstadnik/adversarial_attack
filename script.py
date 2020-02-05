from myimage import MyImage
from myimage_ga import MyImageGA

path = "test.png"

img = MyImageGA(path=path, resize=(1280, -1))
img.compute_data()
img.show()

data = img.data

# print(data)

# img_ = MyImage(img=data['img'][8])
# img_.compute_data()



# %lprun -f img.compute_data img.compute_data()
