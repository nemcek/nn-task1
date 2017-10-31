import queue as q

class EarlyStopping:
    def __init__(self, early_stopping_settings, init_weights):
        self.min_accuracy = early_stopping_settings['min_accuracy']
        self.weights = init_weights
        self.best_epoch = 0
        self.best_error = 100

        self.best_weights = early_stopping_settings['best_weights_delay']
        self.error_accumulated = early_stopping_settings['accumulated_error']
        self.error_raised = early_stopping_settings['raised_error']
        self.error_queue = q.Queue(max(self.error_accumulated['n_previous_epochs'], self.error_raised['n_previous_epochs']))

    def should_stop(self, epoch, error, current_weights, notify=True):
        weights_stop = self.__early_stopping_best_weights_delay(epoch, error, self.best_weights['delay'], current_weights)
        acc_error_stop, raised_error_stop = self.__early_stopping_errors(error)

        if (error <= (100 - self.min_accuracy) / 100):
            if notify:
                if weights_stop:
                    print('Stopping early due to best_weights_delay, Epoch = {:d}, Best Epoch: {:d}'.format(epoch + 1, self.best_epoch + 1))

                if acc_error_stop:
                    print('Stopping early due to accumulated_error, Epoch = {:d}, Best Epoch: {:d}'.format(epoch + 1, self.best_epoch + 1))

                if raised_error_stop:
                    print('Stopping early due to raised_error, Epoch = {:d}, Best Epoch: {:d}'.format(epoch + 1, self.best_epoch + 1))
            return (weights_stop or acc_error_stop or raised_error_stop)
        else:
            return False

    def __early_stopping_best_weights_delay(self, epoch, error, delay, current_weights):
        if error < self.best_error:
            self.weights = current_weights
            self.best_epoch = epoch
            self.best_error = error

        return epoch - self.best_epoch >= delay

    def __early_stopping_errors(self, error):
        if self.error_queue.full():
            self.error_queue.get_nowait()
        self.error_queue.put_nowait(error)

        acc_error = 0
        raised_error = 0
        if self.error_queue.qsize() >= self.error_accumulated['n_previous_epochs']:
            acc_error = sum(self.error_queue.queue[i] - self.error_queue.queue[i - 1] for i in range(1, self.error_accumulated['n_previous_epochs']))

        if self.error_queue.qsize() >= self.error_raised['n_previous_epochs']:
            raised_error = sum(1 if (self.error_queue.queue[i] - self.error_queue.queue[i - 1]) > 0 else 0 for i in range(1, self.error_raised['n_previous_epochs']))

        return acc_error >= self.error_accumulated['threshold'], raised_error >= (self.error_raised['threshold'] * self.error_raised['n_previous_epochs'])