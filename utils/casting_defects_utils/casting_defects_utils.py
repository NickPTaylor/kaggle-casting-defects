"""
Utility functions
"""

import pathlib
import numpy as np
from collections import Counter
from typing import Optional, Dict, Any

from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator

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

def summarise_classes(di: DirectoryIterator) -> Counter:
    """
    Summarise number of classes available from a DirectoryIterator.
    """
    class_lookup = {v: k for k, v in di.class_indices.items()}
    return Counter([class_lookup[cls] for cls in di.classes])
