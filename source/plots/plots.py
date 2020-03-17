"""
Plotting functions.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Any, Optional

import seaborn as sns
from keras.callbacks.callbacks import History

def plot_examples(examples: Dict,
                  output_file: Optional[pathlib.PosixPath] = None,
                  **kwargs: Any) -> None:
    """
    Plot grid of images.
    """

    # Calculate number of columns and rows in grid
    no_rows = len(examples.keys())
    no_cols = np.max([len(examples[k]) for k in examples])

    fig = plt.figure(**kwargs)
    fig.tight_layout()

    # Iterate each class and image and plot the image on a grid.
    for row, (cls, imgs) in enumerate(examples.items()):
        for col, img in enumerate(imgs):
            ax = fig.add_subplot(no_rows, no_cols, (no_cols * row) + (col + 1))
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_axis_off()
            ax.set_title(cls)

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()

def plot_learn(history: History,
               output_file: Optional[pathlib.PosixPath] = None,
               **kwargs: Any) -> None:
    """
    Plot training curve.
    """
    fig, ax = plt.subplots(**kwargs)
    ax.plot(history.history['loss'], label='Training')
    ax.plot(history.history['val_loss'], label='Validation')
    ax.legend()
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str],
                          output_file: Optional[pathlib.PosixPath] = None,
                          **kwargs: Any) -> None:
    """
    Plot the confusion matrix.
    """
    fig, ax = plt.subplots(**kwargs)
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')

    plt.show()

    if output_file is not None:
        plt.savefig(output_file)
