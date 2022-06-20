import keras_tuner as kt
from sklearn import KFold

class CVBayesianTuner(kt.BayesianOptimization):
    """This tuner takes from the bayesian optimisation, but adds cross validation"""

    def __init__(self, *args, n_split=5, **kwargs):
        self.n_split = n_split
        super(CVBayesianTuner, self).__init__(*args, **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        
        
    #     history = self.hypermodel.fit(hp, model, *args, **kwargs)
    #     return

    # def run_trial(self, trial, x, y, **kwargs):

        
    #     KFold(self.n_split).split(training)  # parameter n_split # parameter seed
    #     hp = trial.hyperparameters
    #     model = self.hypermodel.build(hp)
    #     return self.hypermodel.fit(hp, model, *args, **kwargs)

    # def run_trial(self, trial, trial, *args, **kwargs):

    #     original_callbacks = kwargs.pop("callbacks", [])

    #     # Run the training process multiple times.
    #     histories = []
    #     for execution in range(self.executions_per_trial):
    #         copied_kwargs = copy.copy(kwargs)
    #         callbacks = self._deepcopy_callbacks(original_callbacks)
    #         self._configure_tensorboard_dir(callbacks, trial, execution)
    #         callbacks.append(tuner_utils.TunerCallback(self, trial))
    #         # Only checkpoint the best epoch across all executions.
    #         callbacks.append(model_checkpoint)
    #         copied_kwargs["callbacks"] = callbacks
    #         obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)

    #         histories.append(obj_value)
    #     return histories
