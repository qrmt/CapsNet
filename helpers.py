from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def save_results(val_acc, loss, val_loss, file_path):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(val_acc)+1), val_acc)
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(acc)+1),len(acc)/10)
    axs[0].legend(['validation accuracy'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(val_loss)+1),val_loss)
    axs[1].plot(range(1,len(loss)+1),loss)
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(loss)+1),len(loss)/10)
    axs[1].legend(['validation loss', 'train loss'], loc='best')
    plt.savefig(file_path+'.png')
    plt.show()

    # Save results to csv
    results = np.array([np.arange(1,len(loss)+1),loss, val_acc, val_loss])
    results = np.swapaxes(results,0,1)
    df = pd.DataFrame(a, columns=['epoch','training loss','validation accuracy','validation loss'])
    df.to_csv(file_path+'.csv')
