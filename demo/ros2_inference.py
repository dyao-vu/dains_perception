import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import torch

class VLMTrackerNode(Node):
    def __init__(self):
        super().__init__('vlm_tracker')
        self.bridge = CvBridge()
        # Publisher: for your bounding box/tracking output (customize this)
        self.tracks_pub = self.create_publisher(YourDetectionMsgType, 'vlm_tracks', 10)
        # Subscriber: for image topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        # Load your VLM/tracker models here, once
        self.model = ...  # Your load_model(...)
        self.tracker = ...  # Your BYTETracker(...)
        self.get_logger().info("VLM Tracker node ready.")

    def image_callback(self, msg):
        # Convert ROS2 Image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Run inference (insert your preprocessing/inference code here)
        # Use frame.shape for correct sizing
        detections, tracks = your_vlm_tracking_pipeline(frame)
        # Build your output message
        # e.g., custom message: for each track, add (id, x, y, w, h, score)
        detection_msg = YourDetectionMsgType()
        for t in tracks:
            # fill in the details for each track
            ...
        self.tracks_pub.publish(detection_msg)
        # Optional: log or visualize

def main(args=None):
    rclpy.init(args=args)
    node = VLMTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# TrackingDetection.msg
int32 frame_id
int32 object_id
float32 x
float32 y
float32 width
float32 height
float32 score
