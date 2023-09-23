import numpy as np
from PIL import Image

img = Image.open('/Users/amberm/PycharmProjects/MLE/data/img/num_8.png')
img = img.resize((28, 28))
img = img.convert("L")

# img_new = img.point(lambda x: 0 if x > 170 else 1)
# arr = np.array(img_new)
arr = np.array(img)

for i in range(arr.shape[0]):
    print(abs(255-arr[i]))


with open('/CNN/matrix.txt', 'w', encoding='utf-8') as f:
    for i in range(len(arr[0])):
        f.write(f'{list(arr[i])}\n')
        # if i % 27 == 0:
        #     f.write(str(arr[i])+'\n')
