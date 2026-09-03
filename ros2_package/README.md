# GroundingDINO ROS2 Integration

ROS2 package for GroundingDINO object detection and tracking with ByteTrack and CLIPTracker support.

## Features

- ✅ **GroundingDINO detection** - Text-prompted object detection
- ✅ **Dual tracker support** - ByteTrack (fast, IoU-based) and CLIPTracker (appearance-based)
- ✅ **Mission config integration** - Auto-loads class names from mission_briefing/config.json
- ✅ **Custom ROS2 messages** - GroundingDINOTrack and GroundingDINOTrackArray
- ✅ **Visualization output** - Annotated images with track IDs
- ✅ **GPU acceleration** - CUDA support with FP16 option
- ✅ **Docker containerization** - Standalone container for easy deployment

## Architecture

This package wraps the proven `Worker` class from `eval/worker.py`, providing:
- Automatic image resizing (800px short side limit)
- Both ByteTrack and CLIP tracking backends
- CLIP embeddings for appearance-based tracking
- Mission config parsing for dynamic text prompts

## Quick Start

### 1. Build Docker Image

```bash
cd /isis/home/hasana3/vlmtest/GroundingDINO
./docker/build_ros2.sh
```

This builds the `groundingdino_ros:latest` image with:
- ROS2 Humble
- GroundingDINO + dependencies
- ByteTrack + CLIPTracker
- GPU support (CUDA)

### 2. Test Standalone

```bash
# Create ROS2 network
docker network create ros_network_$USER

# Run container
docker run --rm -it --gpus all \
  --network ros_network_$USER \
  -e ROS_DOMAIN_ID=0 \
  groundingdino_ros:latest \
  ros2 topic list
```

### 3. Integration with AirSim

Add the GroundingDINO service to your `docker-compose.yml`:

```yaml
# In airsim/release_installer/docker/docker-compose.yml

services:
  # ... existing services (adk, maneuver, etc.) ...

  groundingdino:
    image: groundingdino_ros:latest
    container_name: groundingdino_${USER}
    networks:
      - ros_network
    environment:
      - ROS_DOMAIN_ID=0
      - ROS_SECURITY_ENABLE=false
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ${PWD}/../../vlmtest/GroundingDINO/weights:/weights:ro
      - ${PWD}/../release/mission_briefing:/mission_briefing:ro
      - ${PWD}/../release/output:/output:rw
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - adk
    command: >
      ros2 launch groundingdino_ros groundingdino.launch.py
      tracker_type:=bytetrack
```

Then start all services:

```bash
cd airsim/release_installer
docker-compose up
```

## ROS2 Topics

### Subscriptions
- `/viaduct/Sim/SceneDroneSensors/robots/Drone1/sensors/front_center1/scene_camera/image` - Input camera images (sensor_msgs/Image)

### Publications
- `/groundingdino/tracks` - Tracking results (GroundingDINOTrackArray)
- `/groundingdino/visualization` - Annotated images with bboxes (sensor_msgs/Image)
- `/groundingdino/debug` - Debug messages (std_msgs/String)

### Switching Trackers

**ByteTrack (default):**
```bash
ros2 launch groundingdino_ros groundingdino.launch.py tracker_type:=bytetrack
```

**CLIPTracker (appearance-based):**
```bash
ros2 launch groundingdino_ros groundingdino.launch.py tracker_type:=clip
```

### Mission Config Format

The node automatically loads class names from `/mission_briefing/config.json`:

```json
{
  "mission": {
    "entities": [
      {"type": "car", "priority": "high"},
      {"type": "pedestrian", "priority": "high"},
      {"type": "drone", "priority": "medium"}
    ]
  }
}
```

This generates the text prompt: `"car. pedestrian. drone."`

## Message Definitions

### GroundingDINOTrack.msg

```
std_msgs/Header header

int32 track_id                  # Unique track ID
int32 frame_id                  # Frame number
string class_name               # Class name
float32 confidence              # Detection score

# Bounding box (pixels)
float32 x, y, width, height

# Bounding box (normalized)
float32 cx_norm, cy_norm, width_norm, height_norm

# Track status
bool is_activated
int32 tracklet_len
```

### GroundingDINOTrackArray.msg

```
std_msgs/Header header
GroundingDINOTrack[] tracks
int32 total_tracks
int32 frame_id
string text_prompt
string tracker_type
```

## Development

### Building Locally (without Docker)

```bash
# Activate ROS2
source /opt/ros/humble/setup.bash

# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ln -s /isis/home/hasana3/vlmtest/GroundingDINO/ros2_package/groundingdino_ros .

# Build
cd ~/ros2_ws
colcon build --packages-select groundingdino_ros

# Source and run
source install/setup.bash
ros2 launch groundingdino_ros groundingdino.launch.py
```

### Testing Messages

```bash
# List topics
ros2 topic list

# Echo tracking results
ros2 topic echo /groundingdino/tracks

# Check message rate
ros2 topic hz /groundingdino/tracks

# Visualize images
ros2 run rqt_image_view rqt_image_view /groundingdino/visualization
```

## Performance

| Tracker | FPS (1080p) | Memory | Use Case |
|---------|-------------|--------|----------|
| ByteTrack | 30-60 | ~4GB | Real-time tracking, clear scenes |
| CLIPTracker | 15-25 | ~6GB | Occlusions, crowded scenes, re-ID |

## Troubleshooting

### No detections
- Lower thresholds: `box_threshold: 0.25`, `text_threshold: 0.25`
- Check text prompt matches objects in scene
- Verify camera topic is publishing: `ros2 topic hz <camera_topic>`

### Track ID switching
- Increase `match_thresh` to 0.9
- Increase `track_buffer` to 60
- Try CLIPTracker for better re-identification

### CUDA out of memory
- Set `use_fp16: true`
- Use SwinT model instead of SwinB
- Reduce image resolution

### Low FPS
- Use ByteTrack instead of CLIPTracker
- Enable FP16: `use_fp16: true`
- Check GPU utilization: `nvidia-smi`

## Files Created

```
ros2_package/groundingdino_ros/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── groundingdino_ros
├── groundingdino_ros/
│   ├── __init__.py
│   ├── groundingdino_node.py      # Main ROS2 node (wraps Worker)
│   └── mission_parser.py          # Mission config parser
├── msg/
│   ├── GroundingDINOTrack.msg
│   └── GroundingDINOTrackArray.msg
├── config/
│   └── default_params.yaml
└── launch/
    └── groundingdino.launch.py

docker/
├── Dockerfile.ros2
├── ros2_entrypoint.sh
├── build_ros2.sh
└── docker-compose.groundingdino.yml
```

## References

- **GroundingDINO Paper**: https://arxiv.org/abs/2303.05499
- **ByteTrack Paper**: https://arxiv.org/abs/2110.06864
- **ROS2 Humble Docs**: https://docs.ros.org/en/humble/
