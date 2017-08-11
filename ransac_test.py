import numpy as np
from skimage.measure import  ransac



t = np.linspace(0, 2 * np.pi, 50)
xc, yc = 20, 30
a, b = 5, 10
x = xc + a * np.cos(t)
y = yc + b * np.sin(t)
data = np.column_stack([x, y])
np.random.seed(seed=1234)
data += np.random.normal(size=data.shape)


data[0] = (100, 100)
data[1] = (110, 120)
data[2] = (120, 130)
data[3] = (140, 130)

model = EllipseModel()
model.estimate(data)
np.round(model.params)



ransac_model, inliers = ransac(data, EllipseModel, 20, 3, max_trials=50)
abs(np.round(ransac_model.params)

inliers
sum(inliers) > 40

from skimage.transform import SimilarityTransform
np.random.seed(0)
src = 100 * np.random.rand(50, 2)
model0 = SimilarityTransform(scale=0.5, rotation=1,translation=(10, 20))
dst = model0(src)
dst[0] = (10000, 10000)
dst[1] = (-100, 100)
>dst[2] = (50, 50)
model, inliers = ransac((src, dst), SimilarityTransform, 2, 10)
inliers
outliers
