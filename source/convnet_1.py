"""
Example of a Convolutional Neural Network.

The data set is from:

    https://www.kaggle.com/ravirajsinh45/
      real-life-industrial-dataset-of-casting-product

and consists of images of castings from a manufacturing process,
labelled according to two categories:

    1) Defective
    2) OK

Details on what 'defective' means are provided in the URL above.  A
training set and test set are provided of 6633 and 715 images
respectively.  The objective is to train a classifier on the test set
and demonstrate its quality by applying it to the test set.
"""

import pathlib

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from source.utils import utils
from source.plots import plots

# Global parameters.
SEED_VALUE = 20200316
BATCH_SIZE = 64

NO_EPOCHS = 5
TRAIN_PROP = 0.05

# Directories for I/O
input_dir = next((pathlib.Path.cwd() / 'data').rglob('input'))
output_dir = next((pathlib.Path.cwd() / 'data').parent.rglob('output'))

# Path of train/test data.
data_sets = ('train', 'test')
train_dir, test_dir = [next(input_dir.rglob(ds)) for ds in data_sets]

# Setup data generators.
trainval_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Show/review example images from generator.
example_images = utils.generate_examples(
    no_examples=5,
    idg=trainval_datagen,
    directory=train_dir,
    target_size=(300, 300),
    color_mode='grayscale',
    seed=SEED_VALUE)
plots.plot_examples(example_images,
                    output_file=output_dir / 'example_images.png',
                    figsize=(8, 4))

# Setup generators for train, validation and test data sets.
train_generator = trainval_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=SEED_VALUE
)

validation_generator = trainval_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED_VALUE
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(300, 300),
    color_mode='grayscale',
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Create neural network architecture.
model = Sequential([
    Conv2D(32, kernel_size=3, input_shape=(300, 300, 1), activation='relu'),
    MaxPool2D(4),
    Conv2D(64, kernel_size=3, input_shape=(300, 300, 1), activation='relu'),
    MaxPool2D(4),
    Conv2D(32, kernel_size=3, input_shape=(300, 300, 1), activation='relu'),
    Flatten(),
    Dense(1, activation='sigmoid')
])
model.summary()

# Compile model.
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Set up callbacks
checkpoint = ModelCheckpoint(
    str(output_dir / 'weights.hdf5'),
    monitor='val_loss'
)
callbacks_list = [checkpoint]

# Train model.
train_steps, val_steps = [TRAIN_PROP * (gen.samples // BATCH_SIZE)
                          for gen in (train_generator, validation_generator)]
training = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_steps,
    validation_data=validation_generator,
    validation_steps=val_steps,
    epochs=NO_EPOCHS,
    callbacks=callbacks_list
)

# Plot training curves.
plots.plot_learn(training, output_file=output_dir / 'training_curve.png')

# Evaluate accuracy.
test_loss, test_acc = model.evaluate_generator(test_generator)
print("Accuracy on test set: {:.3f}".format(test_acc))

# Calculate predicted probabilities, compute confusion matrix, plot.
pred_prob = model.predict_generator(test_generator)
pred_class = [1 if p > 0.5 else 0 for p in pred_prob]
true_class = test_generator.classes
labels = {v: k for k, v in test_generator.class_indices.items()}
labels = [labels[0], labels[1]]

cm = confusion_matrix(pred_class, true_class)

plots.plot_confusion_matrix(cm, labels=labels,
                            output_file=output_dir / 'confusion_matrix.png')
