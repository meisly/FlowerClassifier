import argparse as ap
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import json 
import helper as h




def main(image_path, model_URL, top_k, class_names_file=None):
# Load classnames or assign classnames to none
    if class_names_file:
        with open(class_names_file, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = None
# Load and compile model from file
    model = tf.keras.models.load_model(model_URL, custom_objects={'KerasLayer':hub.KerasLayer})
# Load and process image to be classified
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = h.process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
# Predict and return results
    probabilities = model.predict(processed_image).squeeze()
   
    prob_name_list = h.i_toName(class_names, probabilities)
    
    probs, classes = h.get_topk(prob_name_list, top_k)
    print(probs)
    print(classes)
    return probs, classes



parser = ap.ArgumentParser()
parser.add_argument('image_path', help='enter the path of the file to be classified')
parser.add_argument('model', help='enter the filename of the model to be used.  it must reside in the same directory as the predictor')
parser.add_argument('--topk','-t', type=int, default=3, help='optional. enter the number of predictions to be returned. if no value is entered 3 results will be returned')
parser.add_argument('--name_map', '-n', help='optional. enter a json file to be used to map predictions to names')

args = parser.parse_args()


if __name__ == '__main__':
    main(args.image_path, args.model, args.topk, args.name_map)