## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The calibration images in `camera_cal/` show a chessboard with 9×6 inner corners photographed from 20 different angles and distances.
1. Object points (`find_corners()`): For every image I prepare the same grid of 3D points (0,0,0), (1,0,0), …, (8,5,0). These are the real positions of the corners on a flat board, so z = 0 and the board is assumed to be fixed in space.
2. Image points (`find_corners()`): Each image is converted to grayscale and passed to `cv2.findChessboardCorners()`. When all 54 corners are found, their positions are refined to sub-pixel accuracy with `cv2.cornerSubPix()`. The object/image point pair for that image is then stored.
* Corners were found in 17 of 20 images.
* `calibration1.jpg`, `calibration4.jpg` and `calibration5.jpg` were skipped because part of the board lies outside the frame.

![alt text](chessboard_corners.jpg)

3. Calibration (`calibrate()`): All point pairs are passed to `cv2.calibrateCamera()`. This estimates the camera matrix and the distortion coefficients that best map the ideal grid onto the observed corners.

| Result | Value |
|---|---|
| Image size | 1280 × 720 px |
| Focal length `fx`, `fy` | 	1156.46, 1151.27 px |
| Optical center `cx`, `cy` | 671.32, 389.22 px |
| Distortion `k1`, `k2`, `p1`, `p2`, `k3` | -0.2467, -0.0254, -0.0007, 0.0001, 0.0107 |
| RMS reprojection error | 1.00 px |

The large negative `k1` indicates strong barrel distortion: straight lines bow outward toward the image edges. The tangential terms `p1`, `p2` are close to zero, so the sensor is well aligned with the lens.

The result is saved to `camera_cal/calibration.json` and later loaded with `load_calibration()`, so calibration runs only once.

Applying `cv2.undistort()` (`undistort()`) to `calibration1.jpg` straightens the curved edges of the board. This image was not used for calibration, so it also serves as an independent check:
![alt text](undistort_calibration.png)

### Pipeline (single images)
#### 1. Provide an example of a distortion-corrected image.

The first step of the pipeline undistorts every frame with the stored calibration (`undistort()` in `src/calibration.py`). On a road image the effect is most visible near the image borders. The car hood at the bottom and the white car on the right change shape and position, because the image is "pulled in" at the edges:
![alt text](undistort_test1.png)

Without this step, lane lines that are straight in the world would appear curved. That would bias the lane curvature measured later.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

TODO: Add your text here!!!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

