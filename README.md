# Middleware-Based Distributed Image Processing and Classification System
A complete C++ implementation of a distributed system for medical image classification using deep learning inference.


## Authors

- Md Shamsur Rahman Sami 


## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────┘

     ┌─────────┐                                    ┌─────────┐
     │ Client  │                                    │ Client  │
     │   #1    │                                    │   #N    │
     └────┬────┘                                    └────┬────┘
          │                                              │
          │ Binary Image Data                           │
          │ [size + bytes]                              │
          │                                              │
          └──────────────────┬───────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   MIDDLEWARE   │
                    │   Port: 8080   │
                    │                │
                    │  • Multi-thread │
                    │  • Load balance │
                    │  • Forward req  │
                    └────────┬───────┘
                             │
                             │ Forward Image Data
                             │ [size + bytes]
                             │
                             ▼
                    ┌────────────────┐
                    │     SERVER     │
                    │   Port: 9090   │
                    │                │
                    │  • LibTorch    │
                    │  • OpenCV      │
                    │  • model.pt    │
                    │                │
                    │  Steps:        │
                    │  1. Decode     │
                    │  2. Resize     │
                    │  3. Normalize  │
                    │  4. Inference  │
                    │  5. Classify   │
                    └────────┬───────┘
                             │
                             │ Classification Result
                             │ "COVID-19 (95.3%)"
                             │
                             ▼
                    Return to Client

┌─────────────────────────────────────────────────────────────────┐
│                      DATA FLOW                                   │
└─────────────────────────────────────────────────────────────────┘

Image File → Binary Bytes → Client → Middleware → Server
                                                      ↓
                                                  Decode (OpenCV)
                                                      ↓
                                                  Resize (224x224)
                                                      ↓
                                                  Normalize (ImageNet)
                                                      ↓
                                                  Tensor [1,3,224,224]
                                                      ↓
                                                  Model.forward()
                                                      ↓
                                                  Softmax
                                                      ↓
Result ← Client ← Middleware ← Server ← "Class (confidence%)"
```

## Project Structure

```
project_root/
│
├── client/
│   └── client.cpp              # Client implementation
│
├── middleware/
│   └── middleware.cpp          # Middleware server with threading
│
├── server/
│   ├── server.cpp              # Classification server
│   ├── model_loader.hpp        # Model loader header
│   ├── model_loader.cpp        # Model loader implementation
│   └── model.pt                # Your trained model (TorchScript)
│
├── include/
│   ├── protocol.hpp            # Communication protocol
│   └── utils.hpp               # Utility functions
│
├── build/                      # Build output directory
│   ├── client
│   ├── middleware
│   └── server
│
├── Makefile                    # Build configuration
└── README.md                   # This file
```

## Prerequisites

### 1. LibTorch (PyTorch C++ API)

Download from: https://pytorch.org/

```bash
# Download LibTorch (CPU version)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
sudo mv libtorch /usr/local/

# Or for CUDA version
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
```

### 2. OpenCV

```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

### 3. Build Tools

```bash
sudo apt-get install build-essential pkg-config
```

## Preparing Your Model

### Convert PyTorch Model to TorchScript

```python
import torch
import torchvision.models as models

# Load your trained model
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('E:\CSE 4-2\PDS\Project\Server\best_model.pth'))
model.eval()

# Create example input
example_input = torch.rand(1, 3, 224, 224)

# Convert to TorchScript
traced_model = torch.jit.trace(model, example_input)

# Save
traced_model.save('model.pt')
print("Model saved as model.pt")
```

Place `model.pt` in the `server/` directory.

## Building the Project

### Edit Makefile (If Needed)

Update the LibTorch path in `Makefile`:

```makefile
LIBTORCH_PATH = /usr/local/libtorch  # Change this to your path
```

### Build All Components

```bash
make
```

Or build individually:

```bash
make client
make middleware
make server
```

### Clean Build

```bash
make clean
make rebuild
```

## Running the System

### Step 1: Start the Server

Terminal 1:

```bash
./build/server
# Or specify model path:
./build/server server/model.pt
```

Expected output:
```
[2025-09-26 10:00:00] === Initializing Classification Server ===
[2025-09-26 10:00:00] Loading model from: model.pt
[2025-09-26 10:00:00] CUDA not available, using CPU
[2025-09-26 10:00:00] Model loaded successfully
[2025-09-26 10:00:00] === Classification Server Started ===
[2025-09-26 10:00:00] Listening on port: 9090
[2025-09-26 10:00:00] Waiting for requests...
```

### Step 2: Start the Middleware

Terminal 2:

```bash
./build/middleware
```

Expected output:
```
[2025-09-26 10:00:05] === Middleware Server Started ===
[2025-09-26 10:00:05] Listening on port: 8080
[2025-09-26 10:00:05] Backend server: 127.0.0.1:9090
[2025-09-26 10:00:05] Waiting for clients...
```

### Step 3: Run Client(s)

Terminal 3:

```bash
./build/client "F:\Thesis Dataset\Thesis\Dataset\Lung Disease Dataset\train\Tuberculosis\test_0_14.jpeg"
./build/client "/mnt/f/Thesis Dataset/Thesis/Dataset/Lung Disease Dataset/train/Tuberculosis/test_0_14.jpeg"
./build/client "/mnt/f/Thesis Dataset/Thesis/Dataset/COVID-19_Radiography_Dataset/Lung_Opacity/images/Lung_Opacity-2.png"

```

Or run multiple clients simultaneously:

```bash
# Terminal 3
./build/client images/xray1.jpg

# Terminal 4
./build/client images/xray2.jpg

# Terminal 5
./build/client images/xray3.jpg
```

Expected output:
```
[2025-09-26 10:01:00] === Image Classification Client ===
[2025-09-26 10:01:00] Image: xray1.jpg
[2025-09-26 10:01:00] Reading image: xray1.jpg
[2025-09-26 10:01:00] Image size: 245678 bytes
[2025-09-26 10:01:00] Connecting to middleware at 127.0.0.1:8080
[2025-09-26 10:01:00] Connected successfully
[2025-09-26 10:01:00] Sending image data...
[2025-09-26 10:01:00] Image sent, waiting for classification result...
[2025-09-26 10:01:02] Classification Result: COVID-19 (confidence: 94.56%)

========================================
RESULT: COVID-19 (confidence: 94.56%)
========================================
```

## Configuration

### Ports

Edit the source files to change default ports:

**Client** (`client/client.cpp`):
```cpp
int middlewarePort = 8080;  // Change middleware port
```

**Middleware** (`middleware/middleware.cpp`):
```cpp
int middlewarePort = 8080;  // Port to listen on
int serverPort = 9090;      // Server backend port
```

**Server** (`server/server.cpp`):
```cpp
int serverPort = 9090;      // Port to listen on
```

### Class Names

Edit the class names in `server/server.cpp`:

```cpp
std::vector<std::string> classNames = {
    "Normal",
    "Pneumonia",
    "COVID-19",
    "Tuberculosis",
    "Lung_Cancer",
    // Add your classes here
};
```

### Image Input Size

Edit `model_loader.hpp`:

```cpp
int inputSize = 224;  // Change based on your model
```

## Testing with Multiple Clients

### Bash Script for Stress Testing

Create `test_concurrent.sh`:

```bash
#!/bin/bash

IMAGE_DIR="test_images"
NUM_CLIENTS=10

for i in $(seq 1 $NUM_CLIENTS); do
    IMAGE="${IMAGE_DIR}/image_${i}.jpg"
    echo "Starting client $i with $IMAGE"
    ./build/client "$IMAGE" &
done

wait
echo "All clients completed"
```

Run:
```bash
chmod +x test_concurrent.sh
./test_concurrent.sh
```

## Performance Monitoring

### Server Logs

Monitor server performance:
```bash
./build/server | tee server.log
```

### Middleware Logs

Monitor request distribution:
```bash
./build/middleware | tee middleware.log
```

## Troubleshooting

### LibTorch Not Found

```bash
# Update Makefile with correct path
LIBTORCH_PATH = /path/to/your/libtorch

# Or set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

### Port Already in Use

```bash
# Find process using port
sudo lsof -i :8080
sudo lsof -i :9090

# Kill process
sudo kill -9 <PID>
```

### Model Loading Failed

- Ensure `model.pt` is TorchScript format
- Check model path in server
- Verify model compatibility with LibTorch version

### Connection Refused

- Ensure server is running before middleware
- Ensure middleware is running before client
- Check firewall settings

### Image Decoding Failed

- Ensure image format is supported (JPEG, PNG)
- Check file permissions
- Verify image is not corrupted

## Advanced Features

### Adding Authentication

Modify `protocol.hpp` to add authentication tokens:

```cpp
bool sendAuthenticated(int socket, const std::vector<uint8_t>& data, 
                      const std::string& token) {
    // Send token first, then data
}
```

### Load Balancing Enhancement

Modify `middleware.cpp` to implement round-robin to multiple servers:

```cpp
std::vector<ServerInfo> servers = {
    {"127.0.0.1", 9090},
    {"127.0.0.1", 9091},
    {"127.0.0.1", 9092}
};
```

### Adding Metrics

Track request statistics:

```cpp
struct Metrics {
    std::atomic<int> totalRequests{0};
    std::atomic<int> successfulRequests{0};
    std::atomic<int> failedRequests{0};
    std::atomic<long> totalLatencyMs{0};
};
```

## Performance Benchmarks

Expected performance (on typical hardware):

- **Latency**: 100-500ms per request (depends on model complexity)
- **Throughput**: 10-50 requests/second (with threading)
- **Concurrent Clients**: Tested with up to 100 simultaneous connections

## Contributing

This is a project implementation. For improvements:

1. Add error recovery mechanisms
2. Implement request queuing
3. Add result caching
4. Implement distributed server pools
5. Add monitoring dashboard

## References

- LibTorch Documentation: https://pytorch.org/cppdocs/
- OpenCV Documentation: https://docs.opencv.org/
- Socket Programming: https://beej.us/guide/bgnet/

## License

Academic Project - CSE 4239 Parallel and Distributed Systems

