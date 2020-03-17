def reset_seed(seed: int) -> None:
    """
    Reset seeds for reproducibility.

    This is based on detail in:
        https://github.com/keras-team/keras/issues/
            2743#issuecomment-558216628
        https://machinelearningmastery.com/
            reproducible-results-neural-networks-keras/

    It is observed that this is not perfect but it does result in near
    reproducibility.
    """

    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    import numpy as np
    np.random.seed(seed)

    import random
    random.seed(seed)

    import tensorflow as tf
    tf.random.set_seed(seed)
    session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                              inter_op_parallelism_threads=1,
                                              allow_soft_placement=True,
                                              device_count={'CPU': 1})
    session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                                   config=session_config)
    tf.compat.v1.keras.backend.set_session(session)
