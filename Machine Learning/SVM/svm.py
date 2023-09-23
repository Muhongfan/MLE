import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import _pickle as pickle

mnist = load_digits()
x, test_x, y, test_y = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=40)

model = svm.LinearSVC()
model.fit(x, y)

z = model.predict(test_x)

print('ACC:', np.sum(z == test_y) / z.size)

# with open('./svm.pkl', 'wb') as file:
#     pickle.dump(model, file)

# import _pickle as pickle
# with open('./model.pkl','rb') as file:
#     model = pickle.load(file)
