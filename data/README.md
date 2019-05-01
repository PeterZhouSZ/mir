### Raw samples
You can download the dataset of raw samples from [link](http://cybertron.cg.tu-berlin.de/xiwang/mental_imagery/dataset/images.zip), which contains viewing and recall sequences from 28 observers. 

* File *vi.csv* contains the viewing sequences from observer i, corresponding recall sequences are in file *ri.csv*. For example dataset from the first observer is stored in *v0.csv* and *r0.csv*. 
* Each *csv* file contains 100 eye movement sequences marked by *Image #* at the beginning (e.g. Image 10).
* Each line in one sequence describes one raw sample point with 
	* timestamp in *ms*
	* x coordinate of gaze point in pixel
	* y coordinate of gaze point in pixel

> Note that the eye tracker is calibrated to the screen with resolution of 1920 x 1200. 


### Eye movement events

Detected fixations and saccades in each eye movement sequence can be downloaded from [zip](http://cybertron.cg.tu-berlin.de/xiwang/mental_imagery/dataset/eye_movement_events.zip). Naming convention is the same as for raw sample files.

* Each **fixation** line contains 
	* start timestamp in *ms*
	* end timestamp in *ms*
	* duration in *ms*
	* x coordinate of fixation in pixel
	* y coordinate of fixation in pixel
	* averaged area size of fixation 

* Each **saccade** line contains 
	* start timestamp in *ms*
	* duration in *ms*
	* x coordinate of starting point in pixel
	* y coordinate of starting point in pixel
	* x coordinate of landing point in pixel
	* y coordinate of landing point in pixel
	* amplitude 
	* speed 
