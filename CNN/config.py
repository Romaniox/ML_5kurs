from easydict import EasyDict
from pathlib import Path

cfg = EasyDict()

cfg.TRAIN = EasyDict()
cfg.TRAIN.EPOCHS = 50
cfg.TRAIN.LEARNING_RATE = 0.01
cfg.TRAIN.OPTIMIZER = 'sgd'
cfg.TRAIN.LOAD_MODEL = False
cfg.TRAIN.SAVE_MODEL = True
cfg.TRAIN.EXPERIMENT_NAME = "exp14"

cfg.DATA = EasyDict()
cfg.DATA.ROOT = Path().resolve()
cfg.DATA.NUM_CLASSES = 10
cfg.DATA.TRAIN_BATCH_SIZE = 512
cfg.DATA.TEST_BATCH_SIZE = 64
cfg.DATA.SHUFFLE = True
