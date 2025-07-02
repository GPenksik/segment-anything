import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np
import socket
import pickle
import struct
import multiprocessing.shared_memory as shm

# % Drawing functions taken from onnx_model_example.py
# Draw a mask to an axis as an overlay
def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
# Draw points on an axis
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
# Draw a box on an axis
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

# Define shared memory names
FRAME_SHM_NAME = "SegAnyFrame"
MASK_SHM_NAME = "SegAnyMask"

# Load the video
# video_path = "your_video_vn1_student.mp4"
video_path = "your_video_diar.mp4"
cap = cv2.VideoCapture(video_path)

# Read one frame to determine its shape
ret, frame = cap.read()
if not ret:
    print("APP::init --- Failed to read video frame")
    cap.release()
    exit()

# Convert BGR (OpenCV format) to RGB (Matplotlib format)

# Get frame dimensions
height, width, channels = frame.shape  # RGB frame shape
mask_shape = (height, width)  # Boolean mask shape

# Calculate byte sizes and convert to standard int
frame_size = int(np.prod(frame.shape) * frame.itemsize)
mask_size = int(np.prod(mask_shape) * np.dtype(np.bool_).itemsize)

# Function to check and attach or create shared memory
def get_shared_memory(name, size):
    try:
        # Try to attach to existing shared memory
        shm_obj = shm.SharedMemory(name=name, create=False)
        print(f"APP::get_shared_memory --- Attached to existing shared memory: {name}")
    except FileNotFoundError:
        # Create new shared memory if not found
        shm_obj = shm.SharedMemory(name=name, create=True, size=size)
        print(f"APP::get_shared_memory --- Created new shared memory: {name}")
    return shm_obj

# Get shared memory objects
frame_shm = get_shared_memory(FRAME_SHM_NAME, frame_size)
mask_shm = get_shared_memory(MASK_SHM_NAME, mask_size)

# Create numpy arrays backed by shared memory
frame_shared = np.ndarray(frame.shape, dtype=frame.dtype, buffer=frame_shm.buf)
mask_shared = np.ndarray(mask_shape, dtype=np.bool_, buffer=mask_shm.buf)

# Copy initial data into shared memory
frame_shared[:] = frame  # Copy the RGB frame
mask_shared[:] = False  # Initialize mask with False

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)  # Adjust layout to fit button and input
frame_rgb = cv2.cvtColor(frame_shared, cv2.COLOR_BGR2RGB)

ax.imshow(frame_rgb)
ax.axis('off')  # Hide axes

# Variables for interaction
points = []  # Store multiple points
# Store labels (0 if Ctrl is held, corresponding to image background. 1 otherwise, corresponding to the part of the image we want to keep)
# 2 and 3 are used to store the points for a selection box
point_labels = []  

# This is the frame number that we are currently looking at in the app
frame_number = 1

# This opens a video file, and extracts a requested frame
def get_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("APP::get_frame --- Error: Could not open video.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret = cap.read(frame_shared)
    cap.release()
    
    if not ret:
        print(f"APP::get_frame --- Error: Could not retrieve frame {frame_number}.")
        return None
    
    return ret

# Function to draw points on the image
def draw_points(temp_point=None):
    global frame_rgb
    ax.clear()
    ax.imshow(frame_rgb)
    ax.axis('off')
    
    for (x, y), label in zip(points, point_labels):
        color = 'red' if label == 0 else 'blue'
        # Draw points
        if label in [0, 1]:
                color = 'red' if label == 0 else 'blue'
                ax.plot(x, y, 'o', color=color, markersize=5)
        
        # Draw rectangles between points with label 2 and label 3
        label_2_point = next(((x, y) for (x, y), label in zip(points, point_labels) if label == 2), None)
        label_3_point = next(((x, y) for (x, y), label in zip(points, point_labels) if label == 3), None)
        
        if label_2_point and label_3_point:
            draw_rectangle(label_2_point, label_3_point)
    
    # Draw temporary rectangle if temp_point is provided
    if temp_point and mouse_status.box_start_point:
        draw_rectangle(mouse_status.box_start_point, temp_point, colour='red')
        
    fig.canvas.draw()

def draw_rectangle(label_2_point, label_3_point, colour='blue'):
    x2, y2 = label_2_point
    x3, y3 = label_3_point
    rect_x = min(x2, x3)
    rect_y = min(y2, y3)
    rect_w = abs(x2 - x3)
    rect_h = abs(y2 - y3)
    ax.add_patch(plt.Rectangle((rect_x, rect_y), rect_w, rect_h, edgecolor=colour, facecolor='none', lw=1))

# Variables to store the starting point of the box
class MouseStatus:
    def __init__(self):
        self.box_start_point = None
        self.is_dragging = False
        self.is_pressed = False

mouse_status = MouseStatus()

# Function to handle mouse button press
def on_mouse_press(event):
    global mouse_status

    if event.inaxes != ax:
        return

    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
    
        # # If the user clicks outside of the frame, do nothing
        # if event.xdata is None or event.ydata is None or event.xdata < 0 or event.ydata < 0 or event.xdata >= width or event.ydata >= height:
        #     return

        mouse_status.box_start_point = (x, y)
        mouse_status.is_pressed = True
        print(f"APP::on_mouse_press --- Box start point selected: ({x}, {y}), Label: 2")

# Function to handle mouse motion
def on_mouse_motion(event):
    global mouse_status

    if  mouse_status.is_pressed and event.xdata is not None and event.ydata is not None:
        mouse_status.is_dragging = True
        x, y = int(event.xdata), int(event.ydata)
        draw_points((x, y))
        # print(f"APP::on_mouse_motion --- Mouse dragged to: ({x}, {y})")

# Function to handle mouse button release
def on_mouse_release(event):
    global mouse_status
    
    if event.inaxes != ax:
        return

    if event.xdata is not None and event.ydata is not None:
        # if the user releases the mouse button after dragging, add the box to the list of points
        if mouse_status.is_dragging:
            add_selection_box(event)
        elif not remove_point_if_clicked(event):
            # if a point isnt removed, then process this as a regular point as per the previous implementation in on_press
            add_point(event)

        draw_points()
        mouse_status.is_pressed = False
        mouse_status.is_dragging = False

def add_point(event):
    x, y = int(event.xdata), int(event.ydata)
    ctrl_held = event.key == 'control'  # Check if Ctrl key is held
    label = 0 if ctrl_held else 1
    points.append((x, y))
    point_labels.append(label)
    print(f"APP::on_mouse_release --- Point added: ({x}, {y}), Label: {label}")

def remove_point_if_clicked(event):
    x, y = int(event.xdata), int(event.ydata)
    for i, (px, py) in enumerate(points):
        if abs(px - x) <= 15 and abs(py - y) <= 15:
            del points[i]
            del point_labels[i]
            print(f"APP::remove_point_if_clicked --- Point removed: ({px}, {py})")
            return True
    return False

def add_selection_box(event):
    x, y = int(event.xdata), int(event.ydata)
    if 2 in point_labels and 3 in point_labels:
        index_2 = point_labels.index(2)
        index_3 = point_labels.index(3)
        points[index_2] = mouse_status.box_start_point
        points[index_3] = (x, y)
        print(f"APP::on_mouse_release --- Box points updated: {mouse_status.box_start_point} and ({x}, {y})")
    else:
        points.append(mouse_status.box_start_point)
        point_labels.append(2)
        print(f"APP::on_mouse_release --- Box start point added: {mouse_status.box_start_point}, Label: 2")
        points.append((x, y))
        point_labels.append(3)
        print(f"APP::on_mouse_release --- Box end point added: ({x}, {y}), Label: 3")

    mouse_status.box_start_point = None

# Function to process the currently displayed frame
def process_frame(event):
    global frame_number, ax, points, point_labels, frame_rgb, frame_shared
    
    if not points:
        print("APP::process_frame --- No points or selection box specified.")
        return

    try:
        frame_number = int(text_box.text)  # Get the frame number from input
    except ValueError:
        print(f"APP::process_frame --- Invalid frame number input.")
        return
    
    get_frame(video_path, frame_number) # Get the frame from the video (this also updates shared memory within the function)
    
    # Open socket to communicate with onnx server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 5000))
    
    # Send shared memory name, frame number, points, and labels
    data = pickle.dumps((video_path, frame_number, points, point_labels, frame.shape, frame.dtype.name))
    client_socket.sendall(struct.pack("Q", len(data)) + data)
    
    # Receive processed mask from onnxSegmenter.py onnx server. 
    # This should instead receive confirmation that the mask was calculated
    # The mask itself should also be extracted from shared memory
    data = b""
    payload_size = struct.calcsize("Q")
    while len(data) < payload_size:
        data += client_socket.recv(4096)
    
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    
    while len(data) < msg_size:
        data += client_socket.recv(4096)
    
    mask_frame_size = pickle.loads(data[:msg_size])
    print(f'APP::process_frame --- Received processed info for masks {mask_frame_size}')
    client_socket.close()

    # The mask should be overlayed on the displayed image before points are drawn
    draw_points()  
    if mask_frame_size is not None:
        show_mask(mask_shared, ax)

    fig.canvas.draw()

# Function to update the frame number when the user inputs a new frame number in the text box
def update_frame(text):
    global frame_number, frame_rgb
    try:
        frame_number = int(text)
        new_frame_success = get_frame(video_path, frame_number)
        if new_frame_success:
            frame_rgb = cv2.cvtColor(frame_shared, cv2.COLOR_BGR2RGB)
            draw_points()
    except ValueError:
        print("APP::update_frame --- Invalid frame number input.")

# Create buttons and text box
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, "Process Video")
button.on_clicked(process_frame)

ax_text = plt.axes([0.7, 0.05, 0.1, 0.075])
text_box = TextBox(ax_text, "Frame:", initial=str(frame_number))
text_box.on_submit(update_frame)

# Register mouse event handlers
fig.canvas.mpl_connect("button_press_event", on_mouse_press)
fig.canvas.mpl_connect("button_release_event", on_mouse_release)
fig.canvas.mpl_connect("motion_notify_event", on_mouse_motion)

plt.show()

# Cleanup shared memory when the script exits
frame_shm.close()
frame_shm.unlink()
mask_shm.close()
mask_shm.unlink()