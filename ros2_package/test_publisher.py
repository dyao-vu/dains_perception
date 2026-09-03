#!/usr/bin/env python3
"""
Test image publisher for GroundingDINO node testing.
Publishes video frames as ROS2 Image messages.

Usage:
    python3 test_publisher.py --video /path/to/video.mp4
"""

import argparse
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VideoPublisher(Node):
    """Publishes video frames as ROS2 images for testing."""

    def __init__(self, video_path, fps=10, loop=True):
        super().__init__('video_publisher')

        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(video_path)
        self.loop = loop

        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open video: {video_path}")
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.get_logger().info(f"Video: {self.width}x{self.height}, {self.total_frames} frames")

        # Publisher
        self.image_pub = self.create_publisher(
            Image,
            '/viaduct/Sim/SceneDroneSensors/robots/Drone1/sensors/front_center1/scene_camera/image',
            10
        )

        # Timer for publishing at specified FPS
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.frame_count = 0
        self.get_logger().info(f"Publishing at {fps} FPS to camera topic")

    def timer_callback(self):
        """Publish next frame."""
        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                self.get_logger().info("Video finished, looping...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            else:
                self.get_logger().info("Video finished, stopping.")
                self.timer.cancel()
                return

        if ret:
            # Convert to ROS Image message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f"frame_{self.frame_count}"

            self.image_pub.publish(msg)
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                self.get_logger().info(f"Published frame {self.frame_count}")

    def destroy_node(self):
        """Cleanup."""
        self.cap.release()
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description='Publish video as ROS2 images')
    parser.add_argument('--video', type=str, help='Path to video file', default="../../videos/carla1.mp4")
    parser.add_argument('--fps', type=int, default=30, help='Publishing rate (FPS)')
    parser.add_argument('--no-loop', action='store_true', help='Stop when video ends instead of looping')
    parsed_args = parser.parse_args()

    rclpy.init(args=args)

    try:
        node = VideoPublisher(parsed_args.video, parsed_args.fps, loop=not parsed_args.no_loop)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
