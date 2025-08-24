import cv2
import numpy as np
from scipy.spatial import distance
import logging
import json
from typing import Dict, List, Tuple
from keyframe_primitives.config import Config
from keyframe_primitives.exceptions import KeyframeDetectionError, ExperienceCondensationError
from keyframe_primitives.utils import calculate_optical_flow, calculate_velocity, calculate_keyframe_score
from keyframe_primitives.models import Keyframe, Experience

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeyframePrimitive:
    def __init__(self, config: Config):
        self.config = config
        self.keyframes = []
        self.experience = Experience()

    def keyframe_detection(self, frame: np.ndarray, previous_frame: np.ndarray) -> Keyframe:
        """
        Detects keyframes based on optical flow and velocity.

        Args:
            frame (np.ndarray): Current frame.
            previous_frame (np.ndarray): Previous frame.

        Returns:
            Keyframe: Detected keyframe.
        """
        try:
            optical_flow = calculate_optical_flow(frame, previous_frame)
            velocity = calculate_velocity(optical_flow)
            score = calculate_keyframe_score(velocity, self.config.velocity_threshold)
            if score > self.config.keyframe_threshold:
                keyframe = Keyframe(frame, score)
                return keyframe
            else:
                return None
        except Exception as e:
            logger.error(f"Error detecting keyframe: {str(e)}")
            raise KeyframeDetectionError("Failed to detect keyframe")

    def experience_condensation(self, keyframes: List[Keyframe]) -> Experience:
        """
        Condenses experiences based on keyframes.

        Args:
            keyframes (List[Keyframe]): List of keyframes.

        Returns:
            Experience: Condensed experience.
        """
        try:
            # Sort keyframes by score in descending order
            keyframes.sort(key=lambda x: x.score, reverse=True)

            # Select top N keyframes
            n = self.config.num_keyframes
            keyframes = keyframes[:n]

            # Create condensed experience
            self.experience.keyframes = keyframes
            return self.experience
        except Exception as e:
            logger.error(f"Error condensing experience: {str(e)}")
            raise ExperienceCondensationError("Failed to condense experience")

class KeyframePrimitiveFactory:
    def __init__(self, config: Config):
        self.config = config
        self.keyframe_primitive = KeyframePrimitive(config)

    def create_keyframe_primitive(self) -> KeyframePrimitive:
        return self.keyframe_primitive

def main():
    config = Config()
    factory = KeyframePrimitiveFactory(config)
    keyframe_primitive = factory.create_keyframe_primitive()

    # Load video
    cap = cv2.VideoCapture('path_to_video.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keyframes
        keyframe = keyframe_primitive.keyframe_detection(frame, previous_frame)

        # Condense experience
        if keyframe:
            experience = keyframe_primitive.experience_condensation([keyframe])
            logger.info(f"Detected keyframe with score {keyframe.score}")
            logger.info(f"Condensed experience: {json.dumps(experience.to_dict())}")

        previous_frame = frame

if __name__ == "__main__":
    main()