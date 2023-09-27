import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from matplotlib.animation import FuncAnimation

# The columns we will use to generate our regression model - x,y are the 2 features, z is the predicted continuous output 
xname = "Industry Income Score"
yname= "Teaching Score"
zname= "Research Score"

# Pixels to inches ratio
px=1/96

# Convert them all into ints so we can manipulate them
df = pd.read_csv("unis.csv")[[xname,yname,zname]].applymap(lambda x : int(x))

# The colour map we'll use for the surface
cmap = plt.get_cmap('cool')

# Creating the grid that our model will use on each frame
res = 101
x = np.linspace(0,100,res)
y = np.linspace(0,100,res)
X,Y = np.meshgrid(x,y)

# Initialising our graph
fig, ax = plt.subplots(figsize=(1280*px,1080*px), subplot_kw={"projection" : "3d", "computed_zorder" : False})

pointsperframe = 1

def animate(i):
    
    plt.cla()
    
    n = (i+1)*pointsperframe
    
    this_df = df.head(n)
    
    features = this_df[[xname,yname]].values
    z = this_df[zname].values

    # The model itself. This creates the model but does not train it 
    # I chose a high C value so the plot would reach for the higher up points at the end
    model = SVR(kernel='rbf',C=6)

    # Train the model using 100% of our training data, since the "test" will be graphing the model against the data points
    model.fit(features,z)

    # It's prediction time (grabs crystal ball)
    predictor = lambda x,y : model.predict(np.array([[x,y]])) 

    # Need to format our points in this way so that we can apply the prediction model to them, otherwise numpy will get confused
    points = np.column_stack((X.ravel() , Y.ravel()))

    # Apply the predictor to each point in the grid
    predictions = np.apply_along_axis(lambda row: predictor(row[0], row[1]), axis=1, arr=points)

    # Reshape the predictions to match the shape of X and Y 
    predictions = predictions.reshape(X.shape)

    # GRAPH WORK

    ax.set_title(f"Predicting university research scores using 2 features: n = {n}", fontsize=20)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_zlabel(zname)

    # We'll use the coordinates to plot the actual data points so we can visually observe the difference
    scatter_x = features[:, 0]
    scatter_y = features[:, 1]

    # We're going to plot the graph along with the ORIGINAL scatter points.
    # Due to an error with matplotlib 3D plots, we need to plot the points greater than the predicted and smaller than the predicted separately
    # Therefore let's predict the points themselves and check if they're above or below the surface plot, then plot in order to fix the issue
    predicted_points_on_scatter = np.apply_along_axis(lambda row: predictor(row[0], row[1]), axis=1, arr=features).ravel()

    # Getting points above, points below
    above = z > predicted_points_on_scatter
    below = z <= predicted_points_on_scatter

    # Plot points below first, then plot, then above, which will display in the correct order
    points_below = ax.scatter(scatter_x[below], scatter_y[below], z[below], s=32, edgecolor="black", c="red", label="Original points (overestimate)")

    # Plotting the surface itself - added some alpha so we can see the points more easily
    surface = ax.plot_surface(X,Y,predictions,alpha=0.85, cmap=cmap, label= "SVR-predicted surface",ccount=500,rcount=500)

    # Now finally the points above
    points_above = ax.scatter(scatter_x[above], scatter_y[above], z[above], s=32, edgecolor="black", c="cyan", label="Original points (underestimate)")

    ax.set_xlim(0,100)
    ax.set_ylim(0,100)
    ax.set_zlim(0,100)
    ax.set_zticks(np.linspace(0,100,5))
    
    # Want the corner to start at (0,0)
    ax.invert_xaxis()


    plt.legend()
    plt.savefig(f"frames/{n}.png")   

# After about 1300 samples the plot doesn't change enough to warrant capturing them
for i in range(1305, -1,  -1):
    animate(i)
    print(i)
    

