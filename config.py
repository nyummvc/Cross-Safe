import tensorflow as tf

# (x1_offset, y1_offset, x2_offset, y2_offset)
DEFAULT_BOXES = ((-0.5, -0.5, 0.5, 0.5), (0.2, 0.2, -0.2, -0.2), (-0.8, -0.2, 0.8, 0.2), (-0.2, -0.8, 0.2, 0.8))
NUM_DEFAULT_BOXES = len(DEFAULT_BOXES)

NUM_CLASSES = 2 # blckground class included
NUM_CHANNELS = 3
NUM_PRED_CONF = NUM_DEFAULT_BOXES * NUM_CLASSES
NUM_PRED_LOC  = NUM_DEFAULT_BOXES * 4

IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.2

NEG_POS_RATIO = 5 # pso:neg = 1:5

CONF_THRESH = 0.9


MODEL = "AlexNet"

if MODEL == "AlexNet":
    IMG_H, IMG_W = 300, 400
    FM_SIZES = [[36, 48], [17, 23], [9, 12], [5, 6]]
    

REG_SCALE = 1e-2  # L2 regularization strength
LOC_LOSS_WEIGHT = 1.  # weight of localization loss: loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss


# Training process
RESUME = False  # resume training from previously saved model?
NUM_EPOCH = 1000
BATCH_SIZE = 64  # batch size for training (relatively small)
VALIDATION_SIZE = 0.05  # fraction of total training set to use as validation set
SAVE_MODEL = True  # save trained model to disk?
MODEL_SAVE_PATH = './checkpoint/model.ckpt'  # where to save trained model