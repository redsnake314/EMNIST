import numpy as np
from keras.models import model_from_json
import skimage
import cv2
import imageio
import argparse
import pickle


def load_model(bin_dir):

    # load YAML and create model
    json_file = open('%s/model.json' % bin_dir, 'r')
    loaded_model_yaml = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_yaml)

    # load weights into new model
    model.load_weights('%s/model.h5' % bin_dir)
    return model


def predict_it(model):
    # read parsed image back in 8-bit, black and white mode (L)
    x = cv2.imread('output.png')
    x = np.invert(x)

    # Visualize new array
    imageio.imwrite('resized.png', x)
    x = skimage.transform.resize(x, (28, 28))

    x = x.reshape(-1, 28, 28, 1)

    print(type(x))

    # Convert type to float32
    x = x.astype('float32')

    # Normalize to prevent issues with model
    x /= 255

    # Predict from model

    out = model.predict(x)

    print(out[0])

if __name__ == '__main__':
    # Parse optional arguments
    parser = argparse.ArgumentParser(
        description='A webapp for testing models generated from training.py on the EMNIST dataset')
    parser.add_argument('--bin', type=str, default='bin',
                        help='Directory to the bin containing the model yaml and model h5 files')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='The host to run the flask server on')
    parser.add_argument('--port', type=int, default=5000, help='The port to run the flask server on')
    args = parser.parse_args()

    model = load_model(args.bin)
    mapping = pickle.load(open('%s/mapping.p' % args.bin, 'rb'))
    predict_it(model)
