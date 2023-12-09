from keras.callbacks import Callback
from keras import backend as K

class CustomLearningRateScheduler(Callback):
    def __init__(self, threshold, reduction_factor, min_lr, patience=5):
        super(CustomLearningRateScheduler, self).__init__()
        self.threshold = threshold  # Threshold for validation loss
        self.reduction_factor = reduction_factor  # Factor by which to reduce the LR
        self.min_lr = min_lr  # Minimum learning rate
        self.patience = patience  # Number of epochs to wait before reducing LR
        self.val_loss_history = []  # To track the validation loss history

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        self.val_loss_history.append(current_val_loss)
        current_lr = K.get_value(self.model.optimizer.lr)

        # Check if the current validation loss is below the threshold and if the current learning rate is above the minimum
        if current_val_loss < self.threshold and current_lr > self.min_lr:
            # Check if the validation loss has been on a plateau for the last 'patience' epochs
            if len(self.val_loss_history) >= self.patience and all(loss <= self.threshold for loss in self.val_loss_history[-self.patience:]):
                new_lr = current_lr * self.reduction_factor
                new_lr = max(new_lr, self.min_lr)
                K.set_value(self.model.optimizer.lr, new_lr)
                print(f"\nEpoch {epoch+1}: reducing learning rate to {new_lr}.")
