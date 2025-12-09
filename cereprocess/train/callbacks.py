from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, Specificity, ROC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle


class History:
    def __init__(self, data=None):
        if data == None:
            self.history = {"train": {}, "val": {}}
        else:
            self.history = data
        self.best = {
            "loss": {
                "loss": 1000.0,
            }, 
            "accuracy": {
                "accuracy": -1.0,
            }, 
        }
        self.cm = {
            "actual": [],
            "pred": []
        }
    
    def update_cm(self, actual, pred):
        self.cm["actual"] = actual
        self.cm["pred"] = pred

    def display_cm(self):
        conf = confusion_matrix(self.cm["actual"], self.cm["pred"])
        cm = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=['normal', 'abnormal'])
        cm.plot(cmap = plt.cm.Blues)

    def update(self, metrics, train = 'train'):
        for key, value in metrics.items():
            if key not in self.history[train]:
                self.history[train][key] = []
            self.history[train][key].append(float(value))
        
        if train == 'val':
            if self.best["loss"]["loss"] > float(metrics["loss"]):
                for key, value in metrics.items():
                    self.best["loss"][key] = float(value)
            if self.best["accuracy"]["accuracy"] < float(metrics["accuracy"]):
                for key, value in metrics.items():
                    self.best["accuracy"][key] = float(value)

    def print_best(self):
        print("\nPrinting Best:")
        print("Loss Wise:")
        print(f"Best Loss: {self.best['loss']['loss']}")
        print(self.best["loss"])
        print(f"Best Accuracy: {self.best['accuracy']['accuracy']}")
        print(self.best["accuracy"])
            


    def plot(self, items = None):
        # plot three metrics
        epochs = range(1, len(self.history['val']['loss']) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
       # for i, (key, _) in enumerate(self.history['val'].items()):
       #     if (i > 2):
       #         break
       #     axs[i].plot(self.history['train'][key][:len(epochs)], label='train')
       #     axs[i].plot(self.history['val'][key][:len(epochs)], label='val')
       #     axs[i].set_title(key)
       #     axs[i].set_xlabel('Epochs')
       #     axs[i].legend()
        if items is None:
            items = ['loss', 'f1score', 'accuracy']
        else:
            if len(items) > 3:
                items = items[:3]
        for i, key in enumerate(items):
            axs[i].plot(self.history['train'][key][:len(epochs)], label='train')
            axs[i].plot(self.history['val'][key][:len(epochs)], label='val')
            axs[i].set_title(key)
            axs[i].set_xlabel('Epochs')
            axs[i].legend()
        plt.show()
        
    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.history, file)



class Metrics:
    def __init__(self, metrics):
        self.metrics = metrics

    def update(self, y_true, y_pred):
        for _, metric in self.metrics.items():
            metric.update(y_pred, y_true)

    def compute(self):
        return {key: metric.compute() for key, metric in self.metrics.items()}

    def reset(self):
        for _, metric in self.metrics.items():
            metric.reset()


def def_metrics(device):
    return Metrics({
        'accuracy': Accuracy(task='binary').to(device),
        'precision': Precision(task='binary').to(device),
        'recall': Recall(task='binary').to(device),
        'f1score': F1Score(task='binary').to(device),
        'auroc': AUROC(task='binary').to(device),
        'specificity': Specificity(task='binary').to(device),
        # ROC doesn't work like this
        # 'roc_curve': ROC(task='binary').to(device)
    })
