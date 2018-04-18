
## Advanced Lane Finding Project
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw image.
* Apply a perspective transform to raw images .
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

### Submission Files

* Advanced-Lane-Finding-Project.ipynb is the project solution , all the code contains in it.
* cam_param.p is the camera calibration parameters
* README.md it the writeup file of the project
* project_video_output.mp4 is the performance on the project_video.mp4
* challenge_video_output.mp4 is the performance on the challenge_video.mp4
* harder_challenge_video_output.mp4 is the performance on the harder_challenge_video.mp4
---

### Camera Calibration

#### 1. Camera matrix and distortion coefficients computation and undistort a chessboard

The code for this step is contained in the 2nd code cell of the IPython notebook located in "Advanced-Lane-Finding-Project.ipynbb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 

I then used the pickle package to save the distortion coefficients for the next time. 

finally, I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

[image1]: ./examples/undistort_output.png "Undistorted"
![alt text][image1]


### Pipeline (single images)

#### 1. an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
    
[image2]: ./examples/road_undistorted.png "Road Transformed"
![alt text][image2]

#### 2. Perspective transform description and example.

The code for my perspective transform includes a function called `warp_img(img, M)`, which appears in the 5th code cell of the `Advanced-Lane-Finding-Project.ipynb`.  The `warp_img()` function takes as inputs an image (`img`) and Persperctive Transform Matrix `M`.

the persperctive transform matrix `M` is calculation by cv2.getPerspectiveTransform(src, dst), I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])
```
and I also calculate a inverse perspective transform matrix `Minv` by cv2.getPerspectiveTransform(dst, src) here.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

[image4]: ./examples/warped_straight_lines.png "Warp Example"
![alt text][image4]


#### 3. Thresholded binary image process description and example.
Lots of experiments were conduted in gradient thresholds and the color spaces. As gradient, I have conducted experiment of x dimension, y dimension , magnitude of the gradient and diretion of gradient, the result is shown as below:

[image30]: ./examples/sobel_abs_x.png "Sobel Abs x"
[image31]: ./examples/sobel_abs_y.png "Sobel Abs Y"
[image32]: ./examples/sobel_mag.png "Sobel Mag"
[image33]: ./examples/sobel_dir.png "Sobel Dir"

![alt text][image30]

![alt text][image31]

![alt text][image32]

![alt text][image33]

As the experiments picture shown, absolution gradient in x dimension and magnitude of gradient performance good. and I also experimented in HLS space, as shown below:

[image34]: ./examples/hls_s_ch.png "S channel"
[image35]: ./examples/hls_l_ch.png "L channel"

![alt text][image34]

![alt text][image35]
As the HLS expriments picture shown,in HLS color space, L channel can perform a good in detecting white line. experiments were also done in LAB color space, the result shown that B channel performans good in detecting yellow line. and the result is shown below:

[image36]: ./examples/lab_b_ch.png "B channel"

![alt text][image36]

Ultimately, I used combination of  absolute gradient in x dimension, L channel of HLS and B channel of LAB color spaces, the white line is detected in HLS's L channel and yellow line is detected in LAB's B channel, an OR operation is conducted to combined them together, and the gradient in x dimension is also added to make the detection more robot. thresholding steps is show in function `gradient_color_threhold_pipeline()`.  Here's an example of my output for this step.

[image37]: ./examples/binary_threshold.png "Binary Example"
![alt text][image37]
                                                                                  
                                                                                  

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After applying calibration, thresholding, and a perspective transform to a road image, I get a binary image where the lane lines stand out clearly. `sliding_window_poly_fit_line_finding` and `track_line_finding` are the 2 functions that are used to find lane line pixels and fit the lane position with a 2nd order polynomial. At the first time, I can only blindly search where are the lanes. I first take a histogram along all the columns in the lower half of the image like this:
```python
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
```
and find the bottom base of left lane x and right lane x by taking the largest argument index of left half histogram image and right half histogram image. then divide the image into `nwindows` at row dimension and sliding windows to find the best lane x in each windows.

Ultimately,  I get pixels belonging to left and right lane line , and `polyfit()` method is used to fits a second order polynomial to each set of pixels.The test resul is kinda like this:

[image40]: ./examples/sliding_windows_poly_fit.png "Fit Visual"
![alt text][image40]

Once I have got where the left and right lanes are in the previos frame, In the next frame of video I don't need to do a blind search again, but instead you can just search in a margin around the previous line position, the function `track_line_finding` describe the details and the result like this:

[image41]: ./examples/tracking_lane_finding.png "Tracking Visual"
![alt text][image41]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the pixels is not the real distance of the left and right lane. there is a ratio between pixels and real meter, and the ratio is calculated based on "Radius of Curvature and Distance from Lane Center Calculation" . the code line below shows the calculation:

```python
curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```

`fit[0]` is the y-squared coefficient of the second order polynomial fit, and `fit[1]` is the y coefficient. `y_0` is the y position within the image upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). y_meters_per_pixel is ratio between meters and pixels. 

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:

```python
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * x_meters_per_pix
```

`r_fit_x_int` and `l_fit_x_int` are the bottom x axis intersection point of the right lanes and left lanes respectively. Assuming that the camera is mounted at the center of the vehicle, the car position is the midlle of the image, so the difference between center of intersection points and the image midpoint is the center distance in pixels, and the real distance should mutiply x_meters_per_pix.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

A polygon generated based on left and right fits is fill on the warped image, and the a inverse perspective transfromation is conduted on warped image. the function `fill_lane()` describle it in details. the curve radius and center distance are also draw on the picture. one of the example is shown below:

[image6]: ./examples/final_result.png "Output"
![alt text][image6]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

[video1]: ./project_video_output.mp4 "Video"
Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The chanllenges I faced when implement this project are the change of lightness and shadows etc. some parameters works well on one images but fails at another because of the lightness change or shadows. I investigate the gradient and color spaces to get a robust lane finding, where I found in L channel of HSL color space, I can select a good white lane and B channel in LAB color space performance good in Yellow lane. So I combined them together with a OR operation, Since the color space is sensitive to the light and shadows, I add a constrain in x abs gradient to make it more robust. The model have a good performance on `Project Video.mp4`. but failed in the the harder chanllege video. In the future I want to make the segmentation using Deep Neural Network before detecting line, this may be more robust solution.
