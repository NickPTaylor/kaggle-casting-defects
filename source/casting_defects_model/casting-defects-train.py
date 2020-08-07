import pathlib

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

BATCH_SIZE = 128
NO_EPOCHS = 30
SEED_VALUE = 42

# Paths to input data.
input_dir = pathlib.Path.cwd() / 'input'
data_sets = ('train', 'test')
train_dir, test_dir = [next(input_dir.rglob(ds)) for ds in data_sets]

# Generators.
trainval_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)
test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

# Paths to output data.
output_dir = pathlib.Path().cwd() / 'output'
try:
    output_dir.mkdir()
except FileExistsError:
    pass

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

# Compile network.
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train network.
train_steps, val_steps = [
   (gen.samples // BATCH_SIZE) for
   gen in (train_generator, validation_generator)]

training = model.fit_generator(
   generator=train_generator,
   steps_per_epoch=train_steps,
   validation_data=validation_generator,
   validation_steps=val_steps,
   epochs=NO_EPOCHS
)

#  Save model weights and history.
model.save_weights(output_dir / 'model_weights.h5')
model.save(output_dir / 'full_model.h5')