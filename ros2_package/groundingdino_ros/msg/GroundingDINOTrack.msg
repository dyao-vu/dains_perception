# Single tracked object from GroundingDINO + Tracker

std_msgs/Header header

# Track identification
int32 track_id                  # Unique track ID
int32 frame_id                  # Frame number when this track was seen
string class_name               # Detected class name (e.g., "car", "pedestrian")

# Detection confidence
float32 confidence              # Detection score [0.0, 1.0]

# Bounding box in pixels (top-left corner + dimensions)
float32 x                       # Top-left x coordinate (pixels)
float32 y                       # Top-left y coordinate (pixels)
float32 width                   # Bounding box width (pixels)
float32 height                  # Bounding box height (pixels)

# Bounding box in normalized coordinates [0.0, 1.0]
float32 cx_norm                 # Center x (normalized)
float32 cy_norm                 # Center y (normalized)
float32 width_norm              # Width (normalized)
float32 height_norm             # Height (normalized)

# Track status
bool is_activated               # Whether track is confirmed/activated
int32 tracklet_len              # Number of frames this track has been alive
