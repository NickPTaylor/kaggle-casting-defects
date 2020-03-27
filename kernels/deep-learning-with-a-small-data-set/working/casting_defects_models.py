import pathlib
import pickle
from uuid import uuid4
import numpy as np
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.sequential import Sequential
from keras.models import load_model


class CastingDefectModels:

    def __init__(self,
                 train_dir: pathlib.Path,
                 test_dir: pathlib.Path,
                 output_dir: pathlib.Path,
                 trainval_datagen: ImageDataGenerator,
                 test_datagen: ImageDataGenerator,
                 model: Sequential,
                 no_epochs: int = 5,
                 batch_size: int = 64) -> None:

        self.train_dir = train_dir
        self.trainval_datagen = trainval_datagen
        self.model = model
        self.no_epochs = no_epochs
        self.batch_size = batch_size
        self.train_generator = trainval_datagen.flow_from_directory(
            train_dir,
            target_size=(300, 300),
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
        )
        self.validation_generator = trainval_datagen.flow_from_directory(
            train_dir,
            target_size=(300, 300),
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
        )
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(300, 300),
            color_mode='grayscale',
            batch_size=1,
            class_mode='binary',
            shuffle=False
        )
        self.model_dir = output_dir / 'model'
        self.history_dir = output_dir / 'history'
        self.ensemble_history = None

    def generate_examples(self, nrows: int, ncols: int,
                          random_state: int = 42,
                          figsize: List[float] = [8, 4]) -> plt.Figure:

        """
        Generate a plot of examples of pre-processed images.

        :param nrows: Number of rows of example images per class
        :param ncols: Number of columns of example images per class
        :param random_state: Seed to use for generating images (for
            reproducibility)
        :param figsize: size of figure in inches
        :return: A plt.Figure object.
        """

        # Setup plot.
        fig = plt.figure(figsize=figsize)
        fig.tight_layout()

        # Get list of classes.
        cls_names = [d.name for d in self.train_dir.iterdir()]
        # Iterate classes
        for i, cls in enumerate(cls_names):
            flow_img = self.trainval_datagen.flow_from_directory(
                directory=self.train_dir,
                classes=[cls],
                class_mode=None,
                batch_size=nrows * ncols,
                seed=random_state
            )

            # Iterate images and plot.
            images = next(flow_img)
            for j, img in enumerate(images):
                loc = (i * nrows * ncols) + (j + 1)
                ax = fig.add_subplot(nrows * len(cls_names), ncols, loc)
                ax.imshow(img.squeeze(), cmap='gray')
                ax.set_axis_off()
                ax.set_title(cls)

        return fig

    def train_models(self, no_models: int = 1,
                     save_results: bool = True) -> None:

        """
        Train deep learning models.

        :param no_models: No of models to train.
        :param save_results: Should results be saved?
        """

        # Create directories to save history and models in if the do not exist.
        for d in (self.history_dir, self.model_dir):
            try:
                d.mkdir(parents=True)
            except FileExistsError:
                pass

        # Compile model
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy'])

        # Calculate number of batches required for each epoch.
        train_steps, val_steps = \
            [(gen.samples // self.batch_size)
             for gen in (self.train_generator, self.validation_generator)]

        for i in range(no_models):
            print("Training {} out of {} models...".format(i, no_models))
            history = self.model.fit_generator(
                generator=self.train_generator,
                steps_per_epoch=train_steps,
                validation_data=self.validation_generator,
                validation_steps=val_steps,
                epochs=self.no_epochs,
                verbose=0
            )

            #  Save models and histories if required.
            if save_results:
                fid = uuid4()
                model_fn = self.model_dir / 'model_{}.h5'.format(fid)
                history_fn = self.history_dir / 'history_{}.pickle'.format(fid)

                self.model.save(model_fn)

                with open(str(history_fn), 'wb') as f:
                    pickle.dump(history, f)

    def gather_ensemble_history(self):

        # Generator for history files.
        hist_files = self.history_dir.glob('*.pickle')

        # Initialise dictionary of histories.
        ensemble_hist = dict(
            loss=dict(history=np.empty((0, self.no_epochs))),
            val_loss=dict(history=np.empty((0, self.no_epochs))),
            accuracy=dict(history=np.empty((0, self.no_epochs))),
            val_accuracy=dict(history=np.empty((0, self.no_epochs)))
        )

        # Stack results from each history file to create an array for
        # each metric key of size (no files, no_epochs).
        for hist_file in hist_files:
            with open(str(hist_file), 'rb') as hf:
                history = pickle.load(hf)
                for k, v in history.history.items():
                    new_values = np.expand_dims(np.array(v), 0)
                    ensemble_hist[k]['history'] = \
                        np.vstack([ensemble_hist[k]['history'], new_values])

        # Compute summary statistics for each history.
        for k, v in ensemble_hist.items():
            hist = ensemble_hist[k]['history']
            ensemble_hist[k]['mean'] = \
                np.apply_along_axis(np.mean, 0, hist)
            ensemble_hist[k]['sample_std'] = \
                np.apply_along_axis(np.std, 0, hist, ddof=1)
            ensemble_hist[k]['n'] = \
                np.apply_along_axis(len, 0, hist)
            ensemble_hist[k]['std_err'] = \
                ensemble_hist[k]['sample_std'] / np.sqrt(ensemble_hist[k]['n'])

        self.ensemble_history = ensemble_hist

        return ensemble_hist

    def plot_ensemble_history(self):
        if self.ensemble_history is not None:
            fig, ax = plt.subplots()
            for p in ('loss', 'val_loss'):
                ax.errorbar(x=range(1, self.no_epochs + 1),
                            y=self.ensemble_history[p]['mean'],
                            yerr=1.96 * self.ensemble_history[p]['std_err'],
                            fmt='o-', capsize=5, label=p)
            ax.set_title('Training curve with 95% error bars')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            fig = None

        return fig

    def ensemble_predictions(self):

        # Generator for model files.
        model_files = self.model_dir.glob('*.h5')

        # Stack predictions for each model.
        predictions = []
        i = 0
        for model in model_files:
            i += 1
            print("Predicting with model #{:0>4}: {}".format(i, model.name))
            model = load_model(model)
            predictions.append(model.predict_generator(self.test_generator))
        predictions = np.array(predictions)

        # Compute summary predictions.
        def apply_summary(fun, **kwargs):
            applied_fun = np.apply_along_axis(fun, 0, predictions, **kwargs)
            return list(applied_fun.squeeze())
        summary = dict(
            mean=apply_summary(np.mean),
            sample_std=apply_summary(np.std, ddof=1),
            n=apply_summary(len)
        )
        summary['std_err'] = summary['sample_std'] / np.sqrt(summary['n'])

        return summary, predictions
