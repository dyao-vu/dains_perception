# Array of tracked objects from GroundingDINO + Tracker

std_msgs/Header header

# Tracking results
GroundingDINOTrack[] tracks     # Array of tracked objects

# Summary statistics
int32 total_tracks              # Total number of active tracks in this frame
int32 frame_id                  # Frame number

# Detection metadata
string text_prompt              # Text prompt used for detection
string tracker_type             # Tracker backend used (bytetrack or clip)
