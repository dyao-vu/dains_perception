#!/usr/bin/env python3
"""
Video Saver Node - Subscribes to visualization topic and saves to video file.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import argparse
from pathlib import Path


class VideoSaverNode(Node):
    """Subscribes to image topic and saves frames to video file."""

    def __init__(self, output_path: str, fps: int = 10):
        super().__init__('video_saver')

        self.bridge = CvBridge()
        self.output_path = output_path
        self.fps = fps
        self.writer = None
        self.frame_count = 0

        # Subscribe to visualization topic
        self.image_sub = self.create_subscription(
            Image,
            '/groundingdino/visualization',
            self.image_callback,
            10
        )

        self.get_logger().info(f"Video saver initialized. Output: {output_path}")
        self.get_logger().info("Subscribed to: /groundingdino/visualization")

    def image_callback(self, msg: Image):
        """Save incoming frames to video."""
        try:
            # Convert ROS image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = frame.shape[:2]

            # Initialize video writer on first frame
            if self.writer is None:
                # Use XVID codec - mp4v often has issues with EOF metadata causing infinite loops
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # Change extension to .avi for XVID compatibility
                if self.output_path.endswith('.mp4'):
                    self.output_path = self.output_path.replace('.mp4', '.avi')
                self.writer = cv2.VideoWriter(
                    self.output_path,
                    fourcc,
                    self.fps,
                    (w, h)
                )
                self.get_logger().info(f"Video writer initialized: {w}x{h} @ {self.fps} FPS")

            # Write frame
            self.writer.write(frame)
            self.frame_count += 1

            # Log progress every 30 frames
            if self.frame_count % 30 == 0:
                self.get_logger().info(f"Saved {self.frame_count} frames")

        except Exception as e:
            self.get_logger().error(f"Error saving frame: {e}")

    def __del__(self):
        """Release video writer on shutdown."""
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(f"Video saved: {self.output_path} ({self.frame_count} frames)")


def main():
    parser = argparse.ArgumentParser(description='Save ROS2 image topic to video file')
    parser.add_argument('--output', '-o', required=True, help='Output video file path')
    parser.add_argument('--fps', type=int, default=10, help='Video FPS (default: 10)')
    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()

    try:
        node = VideoSaverNode(str(output_path), args.fps)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.writer is not None:
            node.writer.release()
            print(f"\nVideo saved: {node.output_path} ({node.frame_count} frames)")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
