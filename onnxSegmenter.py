# %%
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Edited by Evguenni Penksik
import pstats
import io
PROFILE = False

if PROFILE:
    import cProfile

def profile_script():
    if PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    # Your existing script code here
    import time
    start_time = time.time()
    # import torch
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    end_time = time.time()
    print(f"Initialisation 1 loading time: {end_time - start_time} seconds")
    start_time = time.time()
    from segment_anything import sam_model_registry, SamPredictor
    end_time = time.time()
    print(f"Initialisation 2 loading time: {end_time - start_time} seconds")
    # from segment_anything.utils.onnx import SamOnnxModel
    start_time = time.time()

    # %
    import onnxruntime
    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization.quantize import quantize_dynamic

    import socket
    import pickle
    import struct

    import subprocess
    import os

    end_time = time.time()
    print(f"Initialisation 3 loading time: {end_time - start_time} seconds")

    start_time = time.time()
    import multiprocessing.shared_memory as shm

    # Start a socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", 5000))
    server_socket.listen(5)

    # Create names
    FRAME_SHM_NAME = "SegAnyFrame"
    MASK_SHM_NAME = "SegAnyMask"

    end_time = time.time()
    print(f"Initialisation 4 loading time: {end_time - start_time} seconds")
    # %%

    def show_mask(mask, ax):
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

    # Set the path below to a SAM model checkpoint, then load the model. This will be needed to both export the model and to calculate embeddings for the model.

    checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    start_time = time.time()
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    end_time = time.time()
    print(f"SAM model loading time: {end_time - start_time} seconds")

    # Assuming `sam` is your model
    # scripted_model = torch.jit.script(sam)
    # torch.jit.save(scripted_model, "sam_scripted.pt")

    # start_time = time.time()
    # sam = torch.jit.load("sam_scripted.pt")
    # end_time = time.time()

    # print(f"SAM model loading time: {end_time - start_time} seconds")

    # %% [markdown]
    # The script `segment-anything/scripts/export_onnx_model.py` can be used to export the necessary portion of SAM. Alternatively, run the following code to export an ONNX model. If you have already exported a model, set the path below and skip to the next section. Assure that the exported ONNX model aligns with the checkpoint and model type set above. This notebook expects the model was exported with the parameter `return_single_mask=True`.

    # %%
    onnx_model_path = "sam_onnx_example.onnx"  # Set to use an already exported model, then skip to the next section.

    image = cv2.imread('images/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Using an ONNX model
    # Here as an example, we use `onnxruntime` in python on CPU to execute the ONNX model. However, any platform that supports an ONNX runtime could be used in principle. Launch the runtime session below:

    start_time = time.time()

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    end_time = time.time()
    print(f"ONNX model loading time: {end_time - start_time} seconds")

    # To use the ONNX model, the image must first be pre-processed using the SAM image encoder. This is a heavier weight process best performed on GPU. SamPredictor can be used as normal, then `.get_image_embedding()` will retreive the intermediate features.
    start_time = time.time()
    sam.to(device='cuda') # Send Sam model to GPU
    predictor = SamPredictor(sam) # Set predictor as SamPredictor
    # predictor.set_image(image) # Set the image used by the predictor
    end_time = time.time()
    print(f"Move model loading time: {end_time - start_time} seconds")

    # Main function to set the image for the predictor and return the first mask prediction
    def set_image_for_predictor(image, points, labels):
        # # The ONNX model has a different input signature than `SamPredictor.predict`. The following inputs must all be supplied. Note the special cases for both point and mask inputs. All inputs are `np.float32`.
        # # * `image_embeddings`: The image embedding from `predictor.get_image_embedding()`. Has a batch index of length 1.
        # # * `point_coords`: Coordinates of sparse input prompts, corresponding to both point inputs and box inputs. Boxes are encoded using two points, one for the top-left corner and one for the bottom-right corner. *Coordinates must already be transformed to long-side 1024.* Has a batch index of length 1.
        # # * `point_labels`: Labels for the sparse input prompts. 0 is a negative input point, 1 is a positive input point, 2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point. *If there is no box input, a single padding point with label -1 and coordinates (0.0, 0.0) should be concatenated.*
        # # * `mask_input`: A mask input to the model with shape 1x1x256x256. This must be supplied even if there is no mask input. In this case, it can just be zeros.
        # # * `has_mask_input`: An indicator for the mask input. 1 indicates a mask input, 0 indicates no mask input.
        # # * `orig_im_size`: The size of the input image in (H,W) format, before any transformation. 
        # # 
        # # Additionally, the ONNX model does not threshold the output mask logits. To obtain a binary mask, threshold at `sam.mask_threshold` (equal to 0.0).
        predictor.set_image(image) # Pre-process image using the SAM image encoder
        
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        image_embedding.shape

        # input_point = np.array([[point_x, point_y]])
        # input_label = np.array([1])

        # input_point = np.array([[point_x, point_y],[point2_x, point2_y]])

        input_points = np.array([[140, 250]])
        input_points = np.concatenate([input_points, np.array([[500, 850]])], axis=0)
        input_points = np.concatenate([input_points, points], axis=0)
        input_labels = np.array([2])
        input_labels = np.concatenate([input_labels, np.array([3])], axis=0)
        input_labels = np.concatenate([input_labels, labels], axis=0)

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        # Package the inputs to run in the onnx model
        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
        }

        # Predict a mask and threshold it.
        masks, _, low_res_logits = ort_session.run(None, ort_inputs)
        masks = masks > predictor.model.mask_threshold

        # Return the first mask in the list
        return masks[0]

    # Function to process the entire video
    def get_frame(video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        cap.release()  # Release resources

        if not ret:
            print(f"Error: Could not retrieve frame {frame_number}.")
            return None
        
        return frame  # Returns the extracted frame
        

    print("ONNX Server is running...")

    if PROFILE:
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        # Print profiling results in a more readable format
        ps.strip_dirs().sort_stats(sortby).print_stats(50)
        # Save profiling results to a file
        with open("profiling_results.txt", "w") as f:
            f.write(s.getvalue())
        # Create a new console window and print the profiling results there
        def print_to_new_console(output):
            # Write the output to a temporary file
            temp_file = "profiling_output.txt"
            with open(temp_file, "w") as f:
                f.write(output)
            
            # Open the temporary file in a new console window
            if os.name == 'nt':  # For Windows
                subprocess.Popen(['start', 'cmd', '/k', 'type', temp_file], shell=True)
            else:  # For Unix-based systems
                subprocess.Popen(['xterm', '-e', 'cat', temp_file])

    # print_to_new_console(s.getvalue())

    while True:
        client_socket, addr = server_socket.accept()
        data = b""

        # Receive the message size first
        payload_size = struct.calcsize("Q")
        while len(data) < payload_size:
            data += client_socket.recv(4096)
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Receive the actual data (video_path, frame_number, points and point_labels)
        while len(data) < msg_size:
            data += client_socket.recv(4096)

        video_path, frame_number, points, point_labels, frame_shape, frame_dtype_name = pickle.loads(data[:msg_size])
        
        print(f"Received request: Video Path = {video_path}, Frame Number = {frame_number}, Start End = {points[0]}:{point_labels[0]}")

        try:
            # Try to load the frame from shared memory
            frame_shm = shm.SharedMemory(name=FRAME_SHM_NAME, create=False)
            frame_size = np.prod(frame_shape) * np.dtype(frame_dtype_name).itemsize
            frame_shared = np.ndarray(frame_shape, dtype=np.dtype(frame_dtype_name), buffer=frame_shm.buf)
            processed_frame = np.copy(frame_shared)
        except FileNotFoundError:
            # If shared memory does not exist, get the frame from the video path
            processed_frame = get_frame(video_path, frame_number)

        frame_shm.close() # Close shared memory
        # Set the image for the predictor and return the first mask prediction
        mask = set_image_for_predictor(processed_frame, points, point_labels)

        # Save the mask to shared memory
        channels, height, width = mask.shape  # shape of the returned mask (should be single channel)
        mask_shape = (height, width)  # Boolean mask shape
        mask_size = (int)(np.prod(mask_shape) * np.dtype(np.bool_).itemsize) # Calcuate the size of the mask
        try:
            mask_shm = shm.SharedMemory(name=MASK_SHM_NAME, create=False, size=mask_size) # Load shared memory. Check if it exists first
        except FileNotFoundError:
            mask_shm = shm.SharedMemory(name=MASK_SHM_NAME, create=True, size=mask_size) # Create shared memory if it does not exist
        mask_shared = np.ndarray(mask_shape, dtype=np.bool_, buffer=mask_shm.buf)
        np.copyto(mask_shared, mask)

        mask_shm.close() # Close shared memory
        
        # Send processed frame info back
        processed_data = pickle.dumps(mask.shape)
        client_socket.sendall(struct.pack("Q", len(processed_data)) + processed_data)
        client_socket.close()
        print("FRAME PROCESSED AND RETURNED")



if __name__ == "__main__":
    profile_script()
