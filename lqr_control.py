"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: template.py << [Modify with your own file name!]

Title: [PLACEHOLDER] << [Modify with your own title]

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: [PLACEHOLDER] << [Write the purpose of the script here]

Expected Outcome: [PLACEHOLDER] << [Write what you expect will happen when you run
the script.]
"""

########################################################################################
# Imports
########################################################################################

import sys

import cv2

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(0, '../library')
import racecar_core, math
import racecar_utils as rc_utils
import numpy as np
import matplotlib.pyplot as plt

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here


########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    pass # Remove 'pass' and write your source code for the start() function here

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    largest_contour, second_largest = find_contours()
    heading, position = get_heading_position()
    K = np.array([0.8742, 0.5])
    angle = compute_steering_angle(heading, position, K)


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass # Remove 'pass and write your source code for the update_slow() function here

def find_contours():
    image = rc.camera.get_color_image()

    color = ((80, 150, 150), (125, 255, 255))  # blue hsv
    CROP_FLOOR = ((200, 0), (rc.camera.get_height(), rc.camera.get_width()))
    MIN_CONTOUR_AREA = 50.0

    largest_contour = None
    largest_area = MIN_CONTOUR_AREA
    second_contour = None
    second_area = MIN_CONTOUR_AREA

    

    if image is not None:
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color[0], color[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                second_area = largest_area
                second_contour = largest_contour
                largest_area = area
                largest_contour = contour
            elif area > second_area:
                second_contour = contour
                second_area = area

    return largest_contour, second_contour

def get_heading_position_line(largest_contour, second_contour):
    if largest_contour is None or second_contour is None:
        return 0.0, 0.0

    M1 = cv2.moments(largest_contour)
    M2 = cv2.moments(second_contour)

    if M1["m00"] == 0 or M2["m00"] == 0: # contour too small
        return 0.0, 0.0

    cx1 = int(M1["m10"] / M1["m00"])
    cy1 = int(M1["m01"] / M1["m00"])

    cx2 = int(M2["m10"] / M2["m00"])
    cy2 = int(M2["m01"] / M2["m00"])

    lane_center_x = (cx1 + cx2) / 2
    lane_center_y = (cy1 + cy2) / 2

    car_center_x = rc.camera.get_width() / 2
    
    # Pos right negative left
    position_offset = car_center_x - lane_center_x

    dx = lane_center_x - car_center_x
    dy = rc.camera.get_height() - 200 - lane_center_y 

    heading_angle = math.atan2(dx, dy)

    return heading_angle, position_offset

def get_heading_position_wall(scan):
    LEFTSTART = 245
    LEFTEND = 330
    RIGHTSTART = 30 
    RIGHTEND = 115
    left_line = get_line(scan, LEFTSTART, LEFTEND)
    right_line = get_line(scan, RIGHTSTART, RIGHTEND)

def get_line(scan, start, end):
    x_array = np.zeros(end-start)
    y_array = np.zeros(end-start)
    for i in range(start, end):
        x = scan[i] * math.cos((i-90) * math.pi / 180)
        y = scan[i] * math.sin((i-90) * math.pi / 180)
        x_array[i-start] = x
        y_array[i-start] = y
    line = np.polyfit(x_array, y_array, 1)
    return line

def compute_steering_angle(heading_error, y_error, K_matrix):
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    x_err = np.array([y_error, heading_error])

    steering_angle = -np.dot(K_matrix, x_err)

    return steering_angle_to_input(steering_angle)


def steering_angle_to_input(steering_angle):
    input = steering_angle / 0.625 # max turn of car
    # print("Steering Angle: ", steering_angle, " Input: ", input)
    return rc_utils.clamp(input, -1, 1)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()