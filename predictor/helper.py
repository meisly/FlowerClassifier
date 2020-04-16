import tensorflow as tf
import numpy as np

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    
    return image.numpy()


def i_toName(namesDict, probAr):
    name_prob_map = []
    if namesDict:
        for i in range(len(probAr)):
            name_prob_map.append((probAr[i], namesDict['{}'.format(i + 1)]))
        name_prob_map.sort(reverse=True)
        return name_prob_map
    else:
        for i in range(len(probAr)):
            name_prob_map.append((probAr[i], i + 1))
        name_prob_map.sort(reverse=True)
        return name_prob_map


def get_topk(probability_name_list, k_num):
    prob_arr = []
    name_arr = []
    for prob, name in probability_name_list[:k_num]:
        prob_arr.append(prob)
        name_arr.append(name)
    return prob_arr, name_arr