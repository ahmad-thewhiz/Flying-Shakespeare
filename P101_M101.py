import cv2
import numpy as np
from PIL import Image

def process_image(image, target_color, tolerance=50):
    width, height = image.size
    pixel_data = image.load()

    for x in range(width):
        for y in range(height):
            r, g, b = pixel_data[x, y]
            tr, tg, tb = target_color

            color_distance = (r - tr)**2 + (g - tg)**2 + (b - tb)**2
            if color_distance <= tolerance**2:
                pixel_data[x, y] = (0, 0, 0)  # Convert color to black
            else:
                pixel_data[x, y] = (255, 255, 255)  # Convert color to white

cap = cv2.VideoCapture(0)

color_lower = np.array([20, 100, 100])
color_upper = np.array([30, 255, 255])

morphology_kernel = np.ones((5, 5), np.uint8)

prev_x, prev_y = -1, -1

line_image = np.ones((480, 640, 3), dtype=np.uint8)
line_image = line_image * 255

while True:
    ret, frame = cap.read()
    frame_shape = frame.shape
    frame = cv2.flip(frame, 1)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_frame, color_lower, color_upper)
    color_mask = cv2.erode(color_mask, morphology_kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, morphology_kernel)
    color_mask = cv2.dilate(color_mask, morphology_kernel, iterations=1)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(largest_contour)

        if prev_x == -1:
            prev_x, prev_y = x + width // 2, y + height // 2
        else:
            line_image = cv2.line(line_image, (prev_x, prev_y), (x + width // 2, y + height // 2), (0, 0, 255), 5)
            frame = cv2.line(frame, (prev_x, prev_y), (x + width // 2, y + height // 2), (255, 255, 255), 5)
            prev_x, prev_y = x + width // 2, y + height // 2
    else:
        prev_x, prev_y = -1, -1

    line_image_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    line_image_inverted = cv2.bitwise_not(line_image_gray)
    white_background = np.full(line_image.shape, 255, dtype=np.uint8)
    background_mask = cv2.bitwise_or(white_background, white_background, mask=line_image_inverted)
    foreground_mask = cv2.bitwise_or(line_image, line_image, mask=line_image_inverted)
    result_image = cv2.bitwise_or(frame, foreground_mask)

    cv2.imshow('Result', result_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('saved_image.png', result_image)
        print("Image saved as 'saved_image.png'")
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Process the saved image to extract the target color
saved_image_path = "saved_image.png"
output_image_path = "output_image.jpg"
target_color = (251, 176, 175)  # Target color in RGB (red in this case)
color_tolerance = 50  # Tolerance for color difference

saved_image = Image.open(saved_image_path)
process_image(saved_image, target_color, color_tolerance)
saved_image.save(output_image_path)
print("Image processed and saved as", output_image_path)
