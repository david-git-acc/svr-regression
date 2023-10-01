import matplotlib.pyplot as plt
import numpy as np

# The goal of this program is to plot the 2 types of errors from the text file errorrect.txt and do some analysis.

f = open("errorrec.txt","r")

# Get some list in string form and convert to a numpy array, too lazy for def since it's only used twice
strlist_to_array = lambda string : np.array([ float(x) for x in string.replace("[","").replace("]","").split(",") ])

# Get the errors as numpy arrays - they're called l1,l2 but I don't think they strictly match the definitions
l1s,l2s = strlist_to_array(f.readline()), np.sqrt( strlist_to_array(f.readline()) )
n = len(l1s)

f.close()

# Pixels to inches ratio
px=1/96
    
# It's plotting time
fig, (axl2,axl1) = plt.subplots(ncols=2,nrows=1,sharex=True, sharey=True,figsize=(1920*px,1080*px))

# X-axis
x = np.arange(1,n+1)

# Initialising the main plots, want lower zorders so they don't cross the error curves
plotl1 = axl1.plot(x,l1s, color="blue",zorder=5)
plotl2 = axl2.plot(x,l2s,color="red",zorder=5)

axl1.set_title("mean absolute difference (blue)",fontsize=20)
axl2.set_title("root mean square difference (red)",fontsize=20)
plt.suptitle(f"Analysing error in the SVR model over the number of training samples (n = 1 to n = {n-1})",fontsize=25)

# Set sensible plot limits so the data is well presented
axl1.set_xlim(0,2341)
axl2.set_xlim(0,2341)
axl1.set_ylim(0,l1s.max()+1)
axl2.set_ylim(0,l2s.max()+1)

axl2.set_ylabel("Error size",fontsize=16)
axl2.set_xlabel("Number of training samples",fontsize=16)

# Plot the minimum and maximum points on the graph so user can see
# The array, the axes to plot on, and if it's a min or a max (False or True)
def plotextreme(arr,ax,min):
    # These gaps were chosen to be convenient to fit on the plot
    xgap,ygap = 150, 0.25
    cutoff=400
    # I chose a cutoff after 400 because otherwise it'd just select the very beginning as the minimum
    xpoint = ( np.argmin( arr[cutoff:] ) if min else np.argmax( arr[cutoff:] )) + cutoff
    ymin = arr[xpoint]

    ax.scatter(xpoint,ymin, s=48, edgecolor="black",color="green",zorder=10)
    ax.text(xpoint-xgap,ymin-ygap,f"({xpoint}, {ymin:.1f})",fontsize=10,zorder=10)

plotextreme(l2s,axl2,True)
plotextreme(l1s,axl1,True)
plotextreme(l2s,axl2,False)
plotextreme(l1s,axl1,False)

# Create a quintic regression model for each 
model1 = np.poly1d(np.polyfit(x, l1s, 5))
model2 = np.poly1d(np.polyfit(x, l2s, 5))

# This function creates a label for the quintic polynomial used in regression for each error plot
# E.g if our quintic regression model has coefficients [-10,-9,8,7,6,-5] it'd return a string label of:
# -1.0e+01x⁵ -9.0e+00x⁴ +8.0e+00x³ +7.0e+00x² +6.0e+00x -5.0e+00
# This is used so that the audience can see the coefficients of the quintic used in regression
def create_label(coefficientlist):
    coefficientsigns = [""] + [" +" if num >= 0 else " " for num in coefficientlist[1:]]
    labelstring=""
    powers = ["⁵","⁴","³","²","",""] # Powers to use
    for index,coeff in enumerate( coefficientlist ):
        # Formatting as a polynomial
        labelstring += f"{coefficientsigns[index]}{coeff:.1e}x{powers[index]}"
    
    # Get rid of the last "x" at the end
    return labelstring[:-1]

# The default ticks were really bad, not enough of them so I'm adding better ones
x_ticks = np.linspace(0,2500,11)       
axl1.set_xticks(x_ticks), axl2.set_xticks(x_ticks) 

# Need more yticks as well
y_ticks = np.arange(0, l2s.max()+1, 0.5)
axl1.set_yticks(y_ticks),axl2.set_yticks(y_ticks)
        
# Plot our quintic regression polynomials, calculated from least squares regression
axl1.plot(x,model1(x), label=f"Regression: {create_label(model1.coefficients)}",zorder=1,color="turquoise")
axl2.plot(x,model2(x), label=f"Regression: {create_label(model2.coefficients)}",zorder=1,color="gold")

# Most empty and clean space to put the legend
fig.legend(loc="lower right")
plt.savefig("errorplot.png")