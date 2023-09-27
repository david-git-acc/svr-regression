# svr-regression

NOTE: due to a tracking error with git, I had to restart the repos from scratch, so there are no logs for this one, sorry. 

This set of programs was made as a way to visually observe the process of SVR machine learning in action - to understand how
the algorithm adapts to the features as more observations are added to the training data. Since visual observation was the 
priority here (so we can gain an intuitive understanding of what it's actually doing), I made 3 design choices:

-I used SVR rather than SVM, as discrete output labels are harder to visually observe compared to a continuous label, which could be represented as an axis (_in this case the z-axis_).
-I only included 2 features, continuous from 0-100% (again for visual clarity), giving 3 dimensions in total, which in addition to a colourmap made it easier for human visualisation
-As specified above I chose a dataset that uses continuous labels, fixed from 0-100% to prevent axis limit changes (_which would add useless space to a plot and hinder the visualisation_)

# how it works

**the plot itself**
The dataset takes a set of 2341 unique universities in the world, takes 3 continuous features (_teaching quality, industry income, research quality_) and then uses SVR to use the former
2 parameters as features to predict the latter. The entire set is used as training data to create the model, which will then use every integer combination of the 2 features (0-100%, 0-100%, 
xy axes) to predict the corresponding research quality (z-axis), which will create a surface plot on how the model predicts the scores. This should hopefully appeal to humans' geometrical
intuition to visually understand how the model makes its predictions. I've also included a colourmap to make it even clearer. 

There is no testing data; the "test" is done by the viewer - the original data points will be plotted on the same set of axes as the plot so it's easy to visually compare the differences between
the predicted output and the actual output. To facilitate this, I've added some alpha transparency to the surface so points both above and below (underestimates and overestimates resp.) can
be seen by the viewer.

**the animation**
In addition, I've also created an animation that over time, iterates through the number of data points used in training the model, from n=125 to n=1232 (_roughly after this point there is no 
significant change in the surface_) so the audience can understand visually how the number of training points influences the behaviour of the model. Also, I like how aesthetically pleasing
it is to watch the surface mold and bend over time. 

# the files 

Note: for all programs, I've increased the number of rows and columns (_rcont,ccont_) interpolated on the surface plot so it appears more smooth and less polygon-y.

**svr_surfaceplot.py** 
This is the program that creates the still image plot. It's what I used as a template to build the animated plot program rather than make everything from scratch.

**svr_surfimgmaker.py**
This is the program that creates the **frames** for the animated plot. I tried to animate them using FuncAnimation but it was too large to efficiently process, so I made it save
each frame instead and make a dedicated program to create the videos.

**videomaker.py**
This program takes the processed frames from the above program and makes the GIF videoclips from them in sections, which can then be combined into a single video. 

**frames** contains all the frames used in the program.
**clips** contains all the clips used in the program.
**mp4s** contains all the separate mp4 videos used to make the final video.

**unis** is the original dataframe.
**mainclip** in both gif and mp4 form (needed gif to post on LinkedIn), is the final animated clip.

The original data can be found from this link:
https://www.kaggle.com/datasets/samiatisha/world-university-rankings-2023-clean-dataset 

