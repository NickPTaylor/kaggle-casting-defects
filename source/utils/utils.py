"""
Utility functions
"""

import pathlib
import numpy as np
from typing import Optional, Dict, Any

from keras_preprocessing.image import ImageDataGenerator

def generate_examples(no_examples: int, idg: ImageDataGenerator,
                      directory: pathlib.PosixPath, seed: Optional[int] = None,
                      **kwargs: Any) -> Dict[str, np.ndarray]:
    """
    Generate examples of pre-processed images.
    """

    # Get list of classes.
    cls_names = [d.name for d in directory.iterdir()]

    # Dictionary of image generators for each class with a batch size
    # equal to no_examples.
    img_generators = dict()
    for cls in cls_names:
        flow_img = idg.flow_from_directory(
            directory=directory,
            classes=[cls],
            class_mode=None,
            batch_size=no_examples,
            seed=seed,
            **kwargs
        )
        img_generators.update({cls: next(flow_img)})

    return img_generators
