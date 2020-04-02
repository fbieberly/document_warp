##############################################################################
#
# Author: Frank Bieberly
# Date: 2 April 2020
# Name: helpers.py
# Description:
# This is a collection of helper functions needed for this project.
#
##############################################################################


from math import sqrt
from itertools import permutations

import numpy as np

def clockwise_order(array):
    # Input: array of 4 2-D points
    # Output: array of 4 2-D points in clockwise order
    #         with the starting point closest to the origin

    if len(array) != 4:
        raise(ValueError, 'Array must be length 4.')
    if any([len(xx) != 2 for xx in array]):
        raise(ValueError, 'Each array element must be a 2D point')


    # get point closest to origin
    array = list(array)
    array.sort(key=lambda xx: sqrt(xx[0]**2+xx[1]**2))

    origin_corner = array[0]

    # generate all permutations of the remaining 3 points
    perm = [xx for xx in permutations(array[1:])]
    ret_arr = []

    # Iterate through the permutations
    for p in perm:

        # Calculate the cross produce of each corner
        # If all cross products are negative, the points are in
        # clockwise order. (in openCV orientation)
        # If all cross products are positive, it is anti-clockwise.
        corner1 = [
                    [origin_corner[0] - p[0][0], origin_corner[1] - p[0][1]],
                    [p[1][0] - p[0][0], p[1][1] - p[0][1]]
                  ]
        corner1_cross = np.cross(corner1[0], corner1[1])

        corner2 = [
                    [p[0][0] - p[1][0], p[0][1] - p[1][1]],
                    [p[2][0] - p[1][0], p[2][1] - p[1][1]]
                  ]
        corner2_cross = np.cross(corner2[0], corner2[1])


        corner3 = [
                    [p[1][0] - p[2][0], p[1][1] - p[2][1]],
                    [origin_corner[0] - p[2][0], origin_corner[1] - p[2][1]]
                  ]
        corner3_cross = np.cross(corner3[0], corner3[1])

        if all([xx < 0 for xx in [corner1_cross, corner2_cross, corner3_cross]]):
            ret_arr = [origin_corner]
            for pt in p:
                ret_arr.append(pt)
            break
    return ret_arr