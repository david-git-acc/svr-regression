import numpy as np
import pandas as pd
from sklearn.svm import SVR

# The goal of this program is to determine the cost function of the SVR model in the animation measured over the training data
# to determine what training data size yields the most optimal model.
# There are 2 cost functions used: mean absolute error and mean square error - we will measure these and record the results on
# errorrec.txt so they can be plotted on errorplotter.py

# The columns we will use to generate our regression model - x,y are the 2 features, z is the predicted continuous output 
xname = "Industry Income Score"
yname= "Teaching Score"
zname= "Research Score"

# Convert them all into ints so we can manipulate them
df = pd.read_csv("unis.csv")[[xname,yname,zname]].applymap(lambda x : int(x))

# Store the error for both types of measurement
l1s = []
l2s = []

# This function is just a cut down version of the animate() function from the animation file
def calcerror(i):
    
    n = i+1
    
    # Get only n samples for use in training data
    this_df = df.head(n)
    
    # Get the features and labels for the data
    features = this_df[[xname,yname]].values
    z = this_df[zname].values

    # The model itself. This creates the model but does not train it 
    # I chose a high C value so the plot would reach for the higher up points at the end
    model = SVR(kernel='rbf',C=6)

    # Train the model using 100% of our training data, since the "test" will be graphing the model against the data points
    model.fit(features,z)

    # It's prediction time (grabs crystal ball)
    predictor = lambda x,y : model.predict(np.array([[x,y]])) 

    # Therefore let's predict the points themselves and check if they're above or below the surface plot, then plot in order to fix the issue
    predicted_points_on_scatter = np.apply_along_axis(lambda row: predictor(row[0], row[1]), axis=1, arr=features).ravel()
    
    # Determine l1 and l2 errors (abs and rms differences)
    l1 = (np.abs(z-predicted_points_on_scatter)).mean()
    l2 = ((z-predicted_points_on_scatter)**2).mean()
    
    l1s.append(l1)
    l2s.append(l2)


# Range of samples to get the error from
L = 2341

# Measuring the error
for i in range(0, L+1,  1):
    calcerror(i)

# Once we have the error recorded, write it down to a file for use in plotting
f = open("errorrec.txt","w")
f.write(str(l1s)+"\n")
f.write(str(l2s))
f.close()