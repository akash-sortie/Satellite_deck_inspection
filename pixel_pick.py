import cv2

# --- PUT YOUR REAL IMAGE PATH HERE ---
image_path = r"C:\Users\HP\Downloads\files\deck_image.jpeg"

# List to store the points you click
clicked_points = []

def get_coordinates(event, x, y, flags, param):
    """Callback function that triggers every time you click the mouse."""
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click
        print(f"[{x}, {y}],")
        clicked_points.append((x, y))
        
        # Draw a small red circle where you clicked so you know it registered
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Pixel Picker", img)

# Load the image
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image at {image_path}")
else:
    print("INSTRUCTIONS:")
    print("1. Click the center of your 4 reference holes.")
    print("2. The coordinates will print in the terminal.")
    print("3. Press the 'q' key on your keyboard to close the window when done.\n")
    
    # Create a window and bind the mouse click function to it
    cv2.namedWindow("Pixel Picker", cv2.WINDOW_NORMAL) # WINDOW_NORMAL lets you resize the window
    cv2.setMouseCallback("Pixel Picker", get_coordinates)

    # Show the image and wait
    cv2.imshow("Pixel Picker", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()