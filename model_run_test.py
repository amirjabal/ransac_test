import numpy as np
from skimage.measure import LineModelND, ransac
from PlaneModelC import PlaneModelND

"""
data1 = np.array([[0, 0, 0],[0,1,0],[1,0,0],[10, 5, 0],[-2, -20, 0],[-30, 30, 0],[50,-10,0],[0,0,3]])
obj1 = mpt.PlaneModelND()
print(obj1.estimate(data1))
print(obj1.params)
print(obj1.residuals(data1))
"""

data1 = np.array([[0, 0, 0],[0,1,0],[1,0,0],[10, 5, 0],[-2, -20, 0],[-30, 30, 0],[50,-10,0],[0,0,3]])


# from lucy's code

for yCoord in range(len(new_blank_image)):
    for xCoord in range(len(new_blank_image[0])):
        x, y, z = depth_to_3d(xCoord, yCoord, P)
        """z = img[yCoord][xCoord]
                                            x = (xCoord - cx) * z / f
                                            y = (yCoord - cy) * z / f"""
        # print(y, x)
        # print("blank_imageyx", blank_image[yCoord][xCoord])
        new_blank_image[yCoord][xCoord] = (x, y, z)






# online, vectorized

def create_point_cloud_vectorized(self,depth_image):
    im_shape = depth_image.shape

    # get the depth
    d = depth_image[:,:,0]

    # replace the invalid data with np.nan
    depth = np.where( (d > 0) & (d < 255), d /256., np.nan)

    # get x and y data in a vectorized way
    row = (np.arange(im_shape[0])[:,None] - self.cx) / self.fx * depth
    col = (np.arange(im_shape[1])[None,:] - self.cy) / self.fy * depth

    # combine x,y,depth and bring it into the correct shape
    return array((row,col,depth)).reshape(3,-1).swapaxes(0,1)






# robustly fit plane and line only using inlier data with RANSAC algorithm
model_robustP, inliersP = ransac(data1, PlaneModelND,  min_samples=3, residual_threshold=1, max_trials=20)
model_robustL1, inliersL1 = ransac(data1, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000)
model_robustL2, inliersL2 = ransac(data1, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000)
