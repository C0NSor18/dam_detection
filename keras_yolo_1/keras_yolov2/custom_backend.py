import sys
#sys.path.append("path/to/backend")
sys.path.append("..")
from keras_yolov2.backend import BaseFeatureExtractor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50


base_path = './backend_weights/custom/'  # FIXME :: use environment variables

RESNET50_BACKEND_PATH = base_path + "resnet50_backend.h5"  # should be hosted on a server


class ResNet50CustomFeature(BaseFeatureExtractor):
    
    def __init__(self, input_size):

        inputs = Input(shape=(input_size[0], input_size[1], 5), name='inputlayer_0')
    
        x = Conv2D(3, (1,1))(inputs)
        resnet50 = ResNet50(weights= 'imagenet', include_top=False)(x)

        self.feature_extractor = Model(inputs=inputs, outputs=resnet50)

        try:
            print("loading backend weights")
            self.feature_extractor.load_weights(RESNET50_BACKEND_PATH)
        except:
            print("Unable to load backend weights. Using a fresh model")
        self.feature_extractor.summary()

