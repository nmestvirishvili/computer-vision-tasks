'''
Group Number: 18
Team Members:
1) Janik Hasche 
2) Christian Kehler
3) Natia Mestvirishvili
'''
#Task2
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt
import numpy as np

#Load Data
with open('noisyedgepoints.npy', 'rb') as f:
   X = np.load(f)
   Y = np.load(f)

#Run RANSAC
ransac = linear_model.RANSACRegressor()
ransac.fit(X, Y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_Y = ransac.predict(line_X)

#Plot Data
plt.figure(figsize=(13, 8))
plt.scatter(X[inlier_mask], Y[inlier_mask], color="black", marker=".", label="Inliers")
plt.scatter(X[outlier_mask], Y[outlier_mask], color="red", marker=".", label="Outliers")
plt.plot(line_X, line_Y, color="cornflowerblue", linewidth = 2, label="RANSAC regressor",)
plt.legend(loc="lower right")
plt.title("RANSAC Edge Model")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
  
