##############################################################################
#
# Author: Frank Bieberly
# Date: 2 April 2020
# Name: document_warp.py
# Description:
# This script takes an input image and will perform a perspective transform
# on a user-defined rectangular region of that image. This code is principally
# designed to warp pictures of paper to give a head-on view of the paper.
# User defines a region by left-clicking 4 points surrounding the region
# they wish to transform. Some automated contour selection will work on high-
# contrast images.
#
# Usage: python perspective_warp.py
#        python perspective_warp.py input_image.jpg
#        python perspective_warp.py input_image.jpg output_image.jpg
#
##############################################################################


import sys
from math import sqrt, floor

import cv2
import numpy as np
from scipy.interpolate import interp1d

from helpers import clockwise_order

# File to open
image_filename = './1.jpg'
output_image_filename = './warp_output.jpg'

# Script can accept a filename command line argument
if len(sys.argv) > 1:
    image_filename = sys.argv[1]
if len(sys.argv) > 2:
    output_image_filename = sys.argv[2]
image_orig = cv2.imread(image_filename)

im_height, im_width = image_orig.shape[:2]
im_ratio = float(im_height)/im_width

# Computer screen width and height (update with your own)
screen_width, screen_height = 1920, 1080

# For portrait mode use 8.5/11.0, else use 11.0/8.5
# (or whatever other dimensions you want)
output_w, output_h = 8.5, 11.0
if im_width > im_height:
    output_w, output_h = output_h, output_w
output_ratio = output_w/output_h
output_height = int(800)
output_width = int(output_height * output_ratio)

# Calculate the resolution to display the image on the screen at
new_width, new_height = 0, 0
if float(im_height)/screen_height > float(im_width)/screen_width:
    new_height = int(screen_height)
    new_width = int(float(new_height)/im_ratio)
else:
    new_width = int(screen_width)
    new_height = int(new_height/im_ratio)

resize_ratio = float(im_width)/new_width
image_resize = cv2.resize(image_orig,(new_width, new_height))
image = image_resize.copy()

# The shape that the warped image will conform to
output_points = [[0, 0],
                 [0, output_height],
                 [output_width, 0],
                 [output_width, output_height]]

# Lists to hold user defined positions on the image
click_list = []
new_contours = []

# Distance to search for nearby contour points in automatic selection
dist_threshold = 150 # distance threshold to find nearby contour points

# Minimum length of a countour to keep
len_thresh = 300

# For grid-warping, set number of polygons per side
# More points can take significantly longer.
x_grid_num = 8
y_grid_num = 8

# Find contours on the users image
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(imgray,(9,9),0)
sigma = 0.30
v = np.median(image)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

edges = cv2.Canny(blur, lower, upper, apertureSize = 3)
kernel = np.ones((2,2), np.uint8)
edges = cv2.dilate(edges, kernel)
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Keep only long contours
long_contours = []
for cnt in contours:
    cnt_len = 0
    for xx in range(len(cnt) -1):
        cnt_len += sqrt((cnt[xx][0][0] - cnt[xx+1][0][0])**2 + (cnt[xx][0][1]- cnt[xx+1][0][1])**2)
    if cnt_len > len_thresh:
        long_contours.append(cnt)

# Mouse event handler. Registers left/right clicks on the image
def onMouse(event, x, y, flags, param):
    global click_list
    global image

    # add contour points (up to 4), deletes the closest point if >= 4.
    if event == cv2.EVENT_LBUTTONDOWN:

        while len(click_list) >= 4:
            dist = 10000000
            min_pt = -1
            for pt in click_list:
                temp_dist = sqrt((pt[0] - x)**2 + (pt[1]-y)**2)
                if temp_dist < dist:
                    dist = temp_dist
                    min_pt = pt
            click_list.remove(min_pt)

        click_list.append([x, y])
        if len(click_list) == 4:
            click_list = clockwise_order(click_list)
        # clears the image and redraws the click_list polygon
        image = image_resize.copy()
        cv2.polylines(image, [np.array(click_list, dtype=np.int32)], True, (0, 0, 255), 2)
    # Right clicking clears the lists
    if event == cv2.EVENT_RBUTTONDOWN:
        click_list = []
        contour_pts = []
        image = image_resize.copy()


# Draw the long contours on the image and display
cv2.drawContours(image, long_contours, -1, (0, 255, 0), 1)
cv2.namedWindow("Original Image");
cv2.setMouseCallback("Original Image", onMouse)
cv2.moveWindow("Original Image", 0,0);
text_box = ['Keyboard short cuts:',
            "'a': automatic selection",
            "'w': warp",
            "'g': grid warp",
            "'s': save"]

# Add some text to remind people of the keyboard shortcuts
for idx, text in enumerate(text_box):
    position = (20, 40+idx*30)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 2) #font stroke

contour_pts = []
contour_dist = 0
while 1:
    cv2.imshow("Original Image", image)
    keypress = ''
    keypress = cv2.waitKey(50)

    # Automatic selection of corner points based on a contour around the target shape
    # Must have selected 4 points near the target corners
    # Draws a contour around the shape if one is found
    if keypress == ord('a'):
        if len(click_list) == 4:
            the_contour = 0
            contour_pts = [[] for xx in click_list]
            contour_dist = [10000000 for xx in click_list]

            # Find the points on the contour nearest to the user-defined
            # click_list points
            distance_violation = False
            for idx, pt in enumerate(click_list):
                for cnt in long_contours:
                    for point in cnt:
                        dist = sqrt((pt[0] - point[0][0])**2 + (pt[1]-point[0][1])**2)
                        if dist < contour_dist[idx]:
                            contour_dist[idx] = dist
                            contour_pts[idx] = list(point[0])
                            the_contour = cnt
                if contour_dist[idx] > dist_threshold:
                    distance_violation = True
                    break

            if distance_violation:
                contour_pts = 0
                print("Cannot find contiguous contour near bounding box.")
                continue
        else:
            print("Need to select 4 points near the corners of the page")

        # Assumption! All the corner points come from the same contour
        the_contour_pts = []
        if contour_pts:
            # Repeat all the points, so there is one big contiguous loop
            for point in the_contour:
                the_contour_pts.append(list(point[0]))
            for point in the_contour:
                the_contour_pts.append(list(point[0]))
        else:
            continue
        # flip them around backwards (so they go clockwise)
        the_contour_pts = the_contour_pts[::-1]

        # Sort the points in clockwise order
        contour_pts = [list(xx) for xx in clockwise_order(contour_pts)]

        # Create 4 new lists of points that represent the top, right, bottom, left
        # edges of the page (in that order)
        new_contours = []
        start_contours = False
        idx = 0
        for pt in the_contour_pts:
            list_pt = list(pt)
            if list_pt == list(contour_pts[0]):

                if idx != 0:
                    new_contours[idx].append(list_pt)
                    break
                new_contours.append([list_pt])
                start_contours = True
            elif not start_contours:
                continue
            elif list_pt in [list(xx) for xx in contour_pts[1:]]:

                new_contours[idx].append(list_pt)
                new_contours.append([list_pt])
                idx += 1
            else:
                new_contours[idx].append(list_pt)

        # Draw the contours on the image, start with dark blue and get lighter each time.
        if len(new_contours) == 4:
            for idx, cnt in enumerate(new_contours):
                cv2.polylines(image, [np.array(cnt, dtype=np.int32)], False, (255, 80*idx, 80*idx), 2)


    # Regular warp
    # If the users has selected 4 click_list points, it will warp those points
    # Additionally, if the user has performed automatic selection,
    # the new contour corner points will be used to warp the image.
    elif keypress == ord('w'):
        if len(click_list) != 4:
            print("Must have four corners selected. {} corners currently selected.".format(len(click_list)))
            continue
        if contour_pts:
            click_list = contour_pts
        output_points = list(output_points)

        # Always need things in clockwise order
        output_points = clockwise_order(output_points)
        click_list = clockwise_order(click_list)

        output_points = np.array(output_points)

        # Create perspective transform matrix
        M = cv2.getPerspectiveTransform(
                    np.array(np.array(click_list)*resize_ratio, dtype=np.float32),
                    np.array(output_points*resize_ratio, dtype=np.float32))

        # warp will be applied to the original (large) image
        warp_input = image_orig.copy()
        warp_result = cv2.warpPerspective(warp_input, M, (int(output_width*resize_ratio), int(output_height*resize_ratio)))

        # But we display a smaller version
        warp_display = cv2.resize(warp_result, (output_width, output_height))

        cv2.namedWindow("Warp Output");
        cv2.moveWindow("Warp Output", 50, 50);
        cv2.imshow("Warp Output", warp_display)

    # Grid warp
    # The users must: select 4 corner points manually, then perform automatic
    # selection to find the contour, then do grid warp. If no contour can be
    # found automatically, grid warp cannot be performed.
    elif keypress == ord('g'): # Grid warp

        if len(new_contours) != 4:
            print("Need to perform automatic selection first (hit 'a')")
            continue
        print("Starting grid warp processing.")

        # Ordered points doesn't work for extreme angles
        output_points = [list(xx) for xx in clockwise_order(output_points)]
        output_points.append(output_points[0])

        # Measure the length of the top/bottom contours to calculate spacing
        # needed in the grid (helpful if the page is heavily skewed)
        top_new_contour_width = sqrt((new_contours[0][-1][0]-new_contours[0][0][0])**2 + \
                                     (new_contours[0][-1][1]-new_contours[0][0][1])**2)
        bot_new_contour_width = sqrt((new_contours[2][-1][0]-new_contours[2][0][0])**2 + \
                                     (new_contours[2][-1][1]-new_contours[2][0][1])**2)
        top_spacing = top_new_contour_width/x_grid_num
        bot_spacing = bot_new_contour_width/x_grid_num

        # Interpolate contour points so they line up equally on the left/right
        # and top/bottom contours
        grid_contours = []
        grid_points = []
        for idx, cnt in enumerate(new_contours):
            cnt_len = 0
            len_vals = [0]

            num = y_grid_num+1
            # idx == 0, 2 are the top/bottom contours
            if idx%2 == 0:
                num = x_grid_num+1
            for xx in range(len(cnt) -1):
                cnt_len += sqrt((cnt[xx][0] - cnt[xx+1][0])**2 + (cnt[xx][1]- cnt[xx+1][1])**2)
                len_vals.append(cnt_len)

            new_len_vals = np.linspace(0, len_vals[-1], num=num)

            # idx == 1 or 3 are the right and left side contours respectively
            if idx == 1:
                new_len_ints = np.linspace(top_spacing, bot_spacing, num=y_grid_num)
                adj_new_len_vals = [0]
                for xx in range(0, y_grid_num):
                    adj_new_len_vals.append(adj_new_len_vals[xx]+new_len_ints[xx])
                ratio = new_len_vals[-1]/adj_new_len_vals[-1]
                new_len_vals = [floor(xx*ratio) for xx in adj_new_len_vals]

            elif idx == 3:
                new_len_ints = np.linspace(bot_spacing, top_spacing, num=y_grid_num)
                adj_new_len_vals = [0]
                for xx in range(0, y_grid_num):
                    adj_new_len_vals.append(adj_new_len_vals[xx]+new_len_ints[xx])
                ratio = new_len_vals[-1]/adj_new_len_vals[-1]
                new_len_vals = [floor(xx*ratio) for xx in adj_new_len_vals]

            # Interpolate at the new length values
            x_vals, y_vals = zip(*cnt)
            f = interp1d(len_vals, x_vals)
            x_vals_new = f(new_len_vals)
            f = interp1d(len_vals, y_vals)
            y_vals_new = f(new_len_vals)
            grid_contours.append([xx for xx in zip(x_vals_new, y_vals_new)])
            for pt in zip(x_vals_new, y_vals_new):
                grid_points.append(pt)

        # Interpolate the output points as well
        output_contours = []
        output_grid_points = []
        for xx in range(len(output_points)-1):
            num = y_grid_num+1
            if xx%2 == 0:
                num = x_grid_num+1
            x_points = np.linspace(output_points[xx][0], output_points[xx+1][0], num=num)
            y_points = np.linspace(output_points[xx][1], output_points[xx+1][1], num=num)
            output_contours.append([ii for ii in zip(x_points, y_points)])
            for pt in zip(x_points, y_points):
                output_grid_points.append([pt[0], pt[1]])

        # Remove a [0, 0] that was appended to output_points (so it made a closed loop)
        output_points = output_points[:-1]

        top_row = grid_contours[0]
        bottom_row = grid_contours[2]
        right_col = grid_contours[1]
        left_col = grid_contours[3]

        output_top_row = output_contours[0]
        output_bottom_row = output_contours[2]
        output_right_col = output_contours[1]
        output_left_col = output_contours[3]

        # Computer the inside grid points so they are spaced equally between top/bottom
        # and left/right sides
        grid_points = []
        output_grid_points = []
        grid_vertices = [[[0, 0] for yy in range(len(left_col))] for xx in range(len(top_row))]
        output_grid_vertices = [[[0, 0] for yy in range(len(left_col))] for xx in range(len(top_row))]
        left_col_length = 0
        right_col_length = 0
        for xx in range(len(left_col)-1):
            left_col_length += sqrt((left_col[xx][0] - left_col[xx+1][0])**2 + (left_col[xx][1]- left_col[xx+1][1])**2)
            right_col_length += sqrt((right_col[xx][0] - right_col[xx+1][0])**2 + (right_col[xx][1]- right_col[xx+1][1])**2)
        for yy in range(len(left_col)):
            for xx in range(len(top_row)):

                ratio = (left_col[y_grid_num - yy][1] - left_col[-1][1])/left_col_length*(x_grid_num-xx)/x_grid_num + (right_col[yy][1] - right_col[0][1])/right_col_length*xx/x_grid_num
                l_ratio = sqrt((left_col[y_grid_num - yy][1] - left_col[-1][1])**2 + (left_col[y_grid_num - yy][0] - left_col[-1][0])**2)/left_col_length*(x_grid_num-xx)/x_grid_num
                r_ratio = sqrt((right_col[yy][1] - right_col[0][1])**2 + (right_col[yy][0] - right_col[0][0])**2)/right_col_length*xx/x_grid_num

                height = sqrt((bottom_row[x_grid_num - xx][0] - top_row[xx][0])**2 + (bottom_row[x_grid_num - xx][1] - top_row[xx][1])**2)
                # height = bottom_row[x_grid_num - xx][1] - top_row[xx][1]

                x_val = (right_col[yy][0] - left_col[y_grid_num - yy][0])/x_grid_num * xx + left_col[y_grid_num - yy][0]
                y_val = (bottom_row[x_grid_num - xx][1] - top_row[xx][1])/y_grid_num * yy + top_row[xx][1]
                y_val = (ratio)*height + top_row[xx][1]

                if yy == 0:
                    y_val = top_row[xx][1]
                elif yy == len(left_col)-1:
                    y_val = bottom_row[x_grid_num - xx][1]
                if xx == 0:
                    x_val = left_col[y_grid_num - yy][0]
                elif xx == len(top_row)-1:
                    x_val = right_col[yy][0]

                grid_vertices[xx][yy] = [x_val, y_val]
                grid_points.append([x_val, y_val])

                x_val = (output_right_col[yy][0] - output_left_col[y_grid_num - yy][0])/x_grid_num * xx + output_left_col[y_grid_num - yy][0]
                y_val = (output_bottom_row[x_grid_num - xx][1] - output_top_row[xx][1])/y_grid_num * yy + output_top_row[xx][1]
                output_grid_vertices[xx][yy] = [x_val, y_val]
                output_grid_points.append([x_val, y_val])

        # Plot the grid points that are inside the contour that is selected
        for pt in grid_points:
            cv2.polylines(image, [np.array([pt], dtype=np.int32)], True, (0, 255, 0), 3)

        # Allocate the warp result image (since we will fill it in iteratively)
        warp_result = np.zeros(np.shape(image_orig[:int(output_height*resize_ratio), :int(output_width*resize_ratio)]), dtype=image.dtype)
        warp_display = np.zeros(np.shape(image[:int(output_height), :int(output_width)]), dtype=image.dtype)

        # For each square in the grid
        for yy in range(len(grid_vertices[0]) - 1):
            for xx in range(len(grid_vertices) - 1):
                temp_input_points = [grid_vertices[xx][yy], grid_vertices[xx][yy+1], grid_vertices[xx+1][yy+1], grid_vertices[xx+1][yy]]
                temp_output_points = [output_grid_vertices[xx][yy], output_grid_vertices[xx][yy+1], output_grid_vertices[xx+1][yy+1], output_grid_vertices[xx+1][yy]]

                temp_output_points_resize = np.array(temp_output_points)*resize_ratio

                # Create perspective transform matrix
                M = cv2.getPerspectiveTransform(
                    np.array(np.array(temp_input_points)*resize_ratio, dtype=np.float32),
                    np.array(temp_output_points_resize, dtype=np.float32))

                temp_output_points_resize = [[int(ii), int(jj)] for ii, jj in temp_output_points_resize]
                # Warp the original image
                result = cv2.warpPerspective(image_orig, M, (int(output_width*resize_ratio), int(output_height*resize_ratio)))

                # Keep only the region that applies the the output region we desire
                region = result[temp_output_points_resize[0][1]:temp_output_points_resize[1][1], temp_output_points_resize[0][0]:temp_output_points_resize[2][0]]
                warp_result[temp_output_points_resize[0][1]:temp_output_points_resize[1][1], temp_output_points_resize[0][0]:temp_output_points_resize[2][0]] = region.copy()

        # Display the smaller resized version
        warp_display = cv2.resize(warp_result, (output_width, output_height))

        cv2.namedWindow("Warp Output")
        cv2.moveWindow("Warp Output", 50, 50)
        cv2.imshow("Warp Output", warp_display)

    # If the Warp Output window exists, save an image of it
    elif keypress == ord('s'):
        if cv2.getWindowProperty("Warp Output",cv2.WND_PROP_AUTOSIZE) >= 1:
            print("Image saved.")
            cv2.imwrite(output_image_filename, warp_result)
        else:
            print("No image found.")
    # Exit the look if Esc is pressed
    elif keypress == 27: # esc
        break
    # Exit loop if the user closes the original image window.
    if cv2.getWindowProperty("Original Image",cv2.WND_PROP_AUTOSIZE) < 1:
        break

cv2.destroyAllWindows()