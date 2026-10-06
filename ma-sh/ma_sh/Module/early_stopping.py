class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        :param patience: Maximum number of epochs to wait without an improvement in the metric.
        :param min_delta: Minimum change in the metric required to count as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.counter = 0
        self.stop_training = False

    def __call__(self, metric):
        # Initialize the best metric with the first observed value
        if self.best_metric is None:
            self.best_metric = metric
            return False

        # Increment the counter if the current metric does not improve sufficiently
        if metric > self.best_metric - self.min_delta:
            self.counter += 1
        else:
            self.best_metric = metric
            self.counter = 0  # Reset the counter when the metric improves

        # Stop training after patience consecutive epochs without improvement
        if self.counter >= self.patience:
            self.stop_training = True

        return self.stop_training
