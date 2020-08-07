import pathlib
import numpy as np

from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator, load_img, \
    img_to_array

# Set up paths.

data_dir = pathlib.Path.cwd() / 'scratch/casting_defects_model'
input_dir = data_dir / 'output'
test_dir = data_dir / 'input/casting_data/test'

# Load model.
model = load_model(input_dir / 'full_model.h5')

# Sanity check using image generator for predictions.
test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(300, 300),
    color_mode='grayscale',
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

test_loss, test_acc = model.evaluate_generator(test_generator)
print(f"Loss: {test_loss:.6f}\nAccuracy: {test_acc:.6f}")

# Sanity check using individual files for predictions.
# NB: Class labels; 0 = 'defective', 1 = 'ok'

def get_pred_from_file(f):
    img = load_img(f, color_mode='grayscale')
    img_arr = img_to_array(img)
    # IMPORTANT - apply same scaling as that applied in model.
    img_arr /= 255.
    img_arr = np.expand_dims(img_arr, axis=0)
    return model.predict(img_arr)[0][0]


preds = []
actuals = []

for img_class in ['def', 'ok']:
    img_files = list((test_dir / f"{img_class}_front").iterdir())
    class_preds = [get_pred_from_file(f) for f in img_files]
    class_actuals = [1 if img_class == 'ok' else 0] * len(class_preds)
    preds += class_preds
    actuals += class_actuals

correct_preds = [np.where(p > 0.5, 1, 0) == a for p, a in zip(preds, actuals)]
print(f"Accuracy: {np.mean(correct_preds):.6f}")
