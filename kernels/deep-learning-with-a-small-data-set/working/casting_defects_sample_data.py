import pathlib
import shutil
import numpy as np

def sample_data(orig_dir: pathlib.Path(), sample_dir: pathlib.Path(),
                no_train: int, no_test: int,
                random_state: int = 42) -> None:
    """
    Generate sample of random data.

    The sample is in the form of symbolic links.  The folder structure
    for the original data is replicated and each data folder contains a
    random sample of symbolic links to the data.
    """

    np.random.seed(random_state)

    # Remove existing sample directory.
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    # Path to test and train directories.
    trn_dir, tst_dir = (next(orig_dir.rglob(ds)) for ds in ('train', 'test'))

    # Iterate test/train and class directories.
    for no_obs, set_dir in zip((no_train, no_test), (trn_dir, tst_dir)):
        for orig_data_dir in set_dir.iterdir():

            # Create data directory, for current set and class.
            # Overwrite if the directory exists.
            rel_dir = orig_data_dir.relative_to(orig_dir)
            class_dir = sample_dir / rel_dir
            class_dir.mkdir(parents=True, exist_ok=True)

            # Get a list of files in original data set and take a sample.
            data_files = list(orig_data_dir.iterdir())
            sample_files = np.random.choice(data_files, size=no_obs,
                                            replace=False)

            # Create a symbolic link to the files.
            for sample_file in sample_files:
                rel_path = sample_file.relative_to(orig_dir)
                symlink_path = class_dir / rel_path.name
                symlink_path.symlink_to(target=sample_file)
