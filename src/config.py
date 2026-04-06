import os
import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)
 
    def __getattr__(self, name):
        # 1. If it's a "magic" method (starts with __), raise AttributeError immediately.
        # This fixes the 'NoneType' is not callable error during pickling.
        if name.startswith('__'):
            raise AttributeError(name)

        # 2. Access internal storage safely
        _inner_dict = self.__dict__.get('_dict')
        
        if _inner_dict is not None and name in _inner_dict:
            return _inner_dict[name]

        # 3. Check the default config
        if name in DEFAULT_CONFIG:
            return DEFAULT_CONFIG[name]

        # 4. For everything else, return None. 
        # This fixes the 'RESULTS' error in HINT.py.
        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


DEFAULT_CONFIG = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'MODEL': 1,                     # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
    'MASK': 3,                      # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
    'NMS': 1,                       # 0: no non-max-suppression, 1: applies non-max-suppression on the external edges by multiplying by Canny
    'SEED': 10,                     # random seed
    'GPU': [0],                     # list of gpu ids
    'AUGMENTATION_TRAIN': 0,        # 1: train 0: false use augmentation to train landmark predictor

    'LR': 0.0001,                   # learning rate
    'D2G_LR': 0.1,                  # discriminator/generator learning rate ratio
    'BETA1': 0.0,                   # adam optimizer beta1
    'BETA2': 0.9,                   # adam optimizer beta2
    'BATCH_SIZE': 4,                # input batch size for training
    'INPUT_SIZE': 256,              # input image size for training 0 for original size
    'MAX_ITERS': 2e6,               # maximum number of iterations to train the model

    'L1_LOSS_WEIGHT': 1,            # l1 loss weight
    'STYLE_LOSS_WEIGHT': 1,         # style loss weight
    'CONTENT_LOSS_WEIGHT': 1,       # perceptual loss weight
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,# adversarial loss weight
    'TV_LOSS_WEIGHT': 0.1,          # total variation loss weight

    'GAN_LOSS': 'lsgan',            # nsgan | lsgan | hinge
    'GAN_POOL_SIZE': 0,             # fake images pool size

    'SAVE_INTERVAL': 1000,          # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 1000,        # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,              # number of images to sample
    'EVAL_INTERVAL': 0,             # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 10,             # how many iterations to wait before logging training status (0: never)

    'USE_LANDMARKS': 1,             # 0: false, 1: true
    'LANDMARK_POINTS': 68,          # 68: celebA, 98: wflw
    'TRAIN_LANDMARK_IMAGE_FLIST': None,
    'TRAIN_LANDMARK_LANDMARK_FLIST': None,
    'VAL_LANDMARK_IMAGE_FLIST': None,
    'VAL_LANDMARK_LANDMARK_FLIST': None,
    'TEST_LANDMARK_IMAGE_FLIST': None,
    'TEST_LANDMARK_LANDMARK_FLIST': None,
}
