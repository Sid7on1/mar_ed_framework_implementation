import logging
import os
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from event_primitives import EventPrimitives
from keyframe_primitives import KeyframePrimitives
from playback_primitives import PlaybackPrimitives
from scipy import spatial

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MAR_ED_Framework:
    """
    Main class for the MAR-ED framework implementation.
    """

    def __init__(self, config: Dict):
        """
        Initialize the MAR-ED framework with the given configuration.

        Args:
        config (Dict): Configuration dictionary containing settings and parameters.
        """
        self.config = config
        self.event_primitives = EventPrimitives()
        self.keyframe_primitives = KeyframePrimitives()
        self.playback_primitives = PlaybackPrimitives()
        self.velocity_threshold = 0.5  # velocity threshold from the paper
        self.flow_theory_threshold = 0.2  # flow theory threshold from the paper

    def main_loop(self):
        """
        Main loop of the MAR-ED framework implementation.
        """
        try:
            # Initialize the scene graph
            self.scene_graph = self.initialize_scene_graph()

            # Capture and store keyframes
            self.capture_keyframes()

            # Playback the experience
            self.experience_playback()

        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")

    def initialize_scene_graph(self) -> Dict:
        """
        Initialize the scene graph with the given configuration.

        Returns:
        Dict: Scene graph dictionary containing nodes and edges.
        """
        try:
            # Initialize the scene graph
            scene_graph = {
                "nodes": [],
                "edges": []
            }

            # Add nodes and edges to the scene graph
            for node in self.config["nodes"]:
                scene_graph["nodes"].append(node)

            for edge in self.config["edges"]:
                scene_graph["edges"].append(edge)

            return scene_graph

        except Exception as e:
            logging.error(f"Error initializing scene graph: {str(e)}")
            return None

    def capture_keyframes(self):
        """
        Capture and store keyframes within the scene graph.
        """
        try:
            # Capture keyframes using the event primitives
            keyframes = self.event_primitives.capture_keyframes()

            # Store keyframes in the scene graph
            self.scene_graph["keyframes"] = keyframes

        except Exception as e:
            logging.error(f"Error capturing keyframes: {str(e)}")

    def experience_playback(self):
        """
        Playback the experience using the keyframes and playback primitives.
        """
        try:
            # Initialize the playback primitives
            self.playback_primitives.initialize_playback()

            # Playback the experience
            for keyframe in self.scene_graph["keyframes"]:
                # Calculate the playback speed based on the user's pace
                playback_speed = self.calculate_playback_speed(keyframe)

                # Playback the keyframe
                self.playback_primitives.playback_keyframe(keyframe, playback_speed)

        except Exception as e:
            logging.error(f"Error in experience playback: {str(e)}")

    def calculate_playback_speed(self, keyframe: Dict) -> float:
        """
        Calculate the playback speed based on the user's pace.

        Args:
        keyframe (Dict): Keyframe dictionary containing the keyframe data.

        Returns:
        float: Playback speed.
        """
        try:
            # Calculate the velocity of the user's movement
            velocity = self.calculate_velocity(keyframe)

            # Calculate the playback speed based on the velocity and flow theory
            playback_speed = self.calculate_playback_speed_from_velocity(velocity)

            return playback_speed

        except Exception as e:
            logging.error(f"Error calculating playback speed: {str(e)}")
            return 1.0

    def calculate_velocity(self, keyframe: Dict) -> float:
        """
        Calculate the velocity of the user's movement.

        Args:
        keyframe (Dict): Keyframe dictionary containing the keyframe data.

        Returns:
        float: Velocity.
        """
        try:
            # Calculate the velocity using the keyframe data
            velocity = spatial.distance.euclidean(keyframe["position"], keyframe["previous_position"])

            return velocity

        except Exception as e:
            logging.error(f"Error calculating velocity: {str(e)}")
            return 0.0

    def calculate_playback_speed_from_velocity(self, velocity: float) -> float:
        """
        Calculate the playback speed based on the velocity and flow theory.

        Args:
        velocity (float): Velocity of the user's movement.

        Returns:
        float: Playback speed.
        """
        try:
            # Calculate the playback speed based on the velocity and flow theory
            if velocity > self.velocity_threshold:
                playback_speed = 1.0 + (velocity - self.velocity_threshold) * self.flow_theory_threshold
            else:
                playback_speed = 1.0

            return playback_speed

        except Exception as e:
            logging.error(f"Error calculating playback speed from velocity: {str(e)}")
            return 1.0

def main():
    # Load the configuration
    config = {
        "nodes": [],
        "edges": []
    }

    # Create an instance of the MAR-ED framework
    mar_ed_framework = MAR_ED_Framework(config)

    # Run the main loop
    mar_ed_framework.main_loop()

if __name__ == "__main__":
    main()