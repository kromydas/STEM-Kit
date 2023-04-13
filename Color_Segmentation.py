import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

def main(lower_bound, upper_bound):
    img = cv2.imread('Emerald_Lakes_New_Zealand_small.jpg')

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    g_lb = np.array([lower_bound, 10, 10], np.uint8)
    g_ub = np.array([upper_bound, 255, 255], np.uint8)

    g_mask = cv2.inRange(img_hsv, g_lb, g_ub)
    g_seg = cv2.bitwise_and(img, img, mask=g_mask)

    output_path = "color_seg_out.jpg"
    cv2.imwrite(output_path, g_seg)
    print(f"Output image path: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the lower and upper bound to segment the green lakes in the image.")
    parser.add_argument("lower_bound", type=int, help="Lower bound for the green color (0 to 255)")
    parser.add_argument("upper_bound", type=int, help="Upper bound for the green color (0 to 255)")

    args = parser.parse_args()

    if 0 <= args.lower_bound <= 255 and 0 <= args.upper_bound <= 255:
        main(args.lower_bound, args.upper_bound)
    else:
        print("Error: Lower and upper bounds must be in the range 0 to 255.")

