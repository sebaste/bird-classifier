"""Provide functionality for classification."""


import logging
import os
import cv2
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

from aux.url_open import url_open, UrlOpenFatalException


class Tf:
    @staticmethod
    def print_info():
        print("Number of available physical GPUs: %d" % len(tf.config.list_physical_devices('GPU')))
        print("Number of available logical GPUs: %d" % len(tf.config.experimental.list_logical_devices('GPU')))

    @staticmethod
    def init(tfhub_cache_dir):
        os.environ["TFHUB_CACHE_DIR"] = os.path.expanduser(tfhub_cache_dir)


class ClassificationFatalException(Exception):
    """Exception signifying a fatal error in classification."""


class ClassificationResult:
    """Classification result, containing name and probability."""

    def __init__(self, name, probability):
        self.name = name
        self.probability = probability
    def __str__(self):
        return "({}, {})".format(self.name, self.probability)


class Classification:
    """Namespace for functionality used for classification."""

    @staticmethod
    def load_model(url_model):
        """Load a KerasLayer model from a given URL."""

        model = None
        try:
            model = hub.KerasLayer(url_model)
            if not model:
                logging.warning("No model returned for URL '%s'", url_model)
                raise
        except Exception:
            raise ClassificationFatalException
        return model

    @staticmethod
    def load_labels(url_labels):
        """Load labels from a given URL."""

        labels_raw = None
        try:
            labels_raw = url_open(url_labels)
        except UrlOpenFatalException:
            raise ClassificationFatalException
        labels_lines = [line.decode('utf-8').replace('\n', '') for line in labels_raw.readlines()]
        labels_lines.pop(0) # Remove header (id, name).
        labels = {}
        for line in labels_lines:
            e_id = int(line.split(',')[0])
            e_name = line.split(',')[1]
            labels[e_id] = {'name': e_name}
        return labels

    @staticmethod
    def order_by_result_score(model_raw_output, labels):
        """Order model output based on score values."""

        for index, value in np.ndenumerate(model_raw_output):
            index = index[1]
            labels[index]['score'] = value
        return sorted(labels.items(), key=lambda x: x[1]['score'])

    @staticmethod
    def get_top_n_result(top_index, names_with_results_ordered):
        """Get the top n results from an ordered list."""

        name = names_with_results_ordered[top_index*(-1)][1]['name']
        score = names_with_results_ordered[top_index*(-1)][1]['score']
        return ClassificationResult(name, score)

    @staticmethod
    def load_image(image):
        """Load an image from a URL."""

        image_get_response = None
        try:
            image_get_response = url_open(image)
        except UrlOpenFatalException:
            raise ClassificationFatalException
        return np.asarray(bytearray(image_get_response.read()), dtype=np.uint8)

    @staticmethod
    def format_image(image_array):
        """Format an image."""

        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image / 255

    @staticmethod
    def generate_tensor(image):
        """Generate a tensor for an image."""

        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        return tf.expand_dims(image_tensor, 0)
