# Modified from: https://www.tensorflow.org/tutorials/keras/basic_classification

import matplotlib.pyplot as plt
import numpy as np


class Results:

    @staticmethod
    def plot_image(i, predictions_array, true_label, img, category_names):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% ({})".format(category_names[predicted_label],
                                             100 * np.max(predictions_array),
                                             category_names[true_label]),
                   color=color)

    @staticmethod
    def plot_value_array(i, predictions_array, true_label, n_categories):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        this_plot = plt.bar(range(n_categories), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        this_plot[predicted_label].set_color('red')
        this_plot[true_label].set_color('blue')
