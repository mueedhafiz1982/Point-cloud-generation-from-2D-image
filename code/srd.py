import numpy as np
#from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1,
        'summaryLoss': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    training_accuracies =   event_acc.summaryLoss('summaryLoss')
    #validation_accuracies = event_acc.Scalars('validation_accuracy')
"""
    steps = 10
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in xrange(steps):

        y[i, 0] = loss[i][2] # value
    plt.plot(x, y[:,0], label='training accuracy')
        y[i, 0] = training_accuracies[i][2] # value
        y[i, 1] = validation_accuracies[i][2]

    plt.plot(x, y[:,0], label='training accuracy')
    plt.plot(x, y[:,1], label='validation accuracy')    
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()
"""    

if __name__ == '__main__':
    log_file = "/content/drive/MyDrive/attprj1/3D-point-cloud-generation/summary_0/orig-ft/events.out.tfevents.1619896749.864d7740a053"
    plot_tensorflow_log(log_file)
