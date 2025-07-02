import onnxruntime as ort
import numpy as np
import cv2
import socket
import pickle
import struct

# Load ONNX model once in memory
onnx_model_path = "sam_onnx_example.onnx"
session = ort.InferenceSession(onnx_model_path)

# Start a socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("localhost", 5000))
server_socket.listen(5)

print("ONNX Server is running...")

while True:
    client_socket, addr = server_socket.accept()
    data = b""
    
    # Receive the frame size first
    payload_size = struct.calcsize("Q")
    while len(data) < payload_size:
        data += client_socket.recv(4096)
    
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    # Receive the frame data
    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    frame = pickle.loads(frame_data)

    # Process frame using ONNX model
    input_tensor = np.expand_dims(frame, axis=0).astype(np.float32)
    outputs = session.run(None, {"input": input_tensor})
    processed_frame = outputs[0].squeeze().astype(np.uint8)

    # Send processed frame back
    processed_data = pickle.dumps(processed_frame)
    client_socket.sendall(struct.pack("Q", len(processed_data)) + processed_data)
    client_socket.close()
