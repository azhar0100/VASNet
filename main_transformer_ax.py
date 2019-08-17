from main_transformer import *

import numpy as np

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.metrics.branin import branin
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

fixed_d = {
    "splits" : ['splits/tvsum_aug_splits_quick.json',
                    'splits/summe_aug_splits_quick.json']
}


print_pkg_versions()
def eval_s(d):
    hps = HParameters()

    print("Checking {} now".format(d))

    d.update(fixed_d)
    d['lr'] = [d['lr']]
    d['n_heads'] = 2**d['n_heads']
    hps.literal_load_from_args(d)

    print("Parameters:")
    print("----------------------------------------------------------------------")
    print(hps)

    scores = []

    if hps.train:
        train(hps)
    else:
        results=[['No', 'Split', 'Mean F-score']]
        for i, split_filename in enumerate(hps.splits):
            f_score = eval_split(hps, split_filename, data_dir=hps.output_dir)
            scores.append(f_score)
            results.append([i+1, split_filename, str(round(f_score * 100.0, 3))+"%"])

        print("\nFinal Results:")
        print_table(results)

    return {"f_score_tvsum": (scores[0], 0.0), "f_score_summe": (scores[1], 0.0) }

best_parameters, values, experiment, model = optimize(
    parameters=[
        {
            "name": "n_heads",
            "type": "range",
            "bounds": [0,3],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        },
        {
            "name": "l2_req",
            "type": "range",
            "bounds": [0.0, 0.1],
        },
        {
            "name": "lr",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "epochs_max",
            "type": "range",
            "bounds": [0, 500],
        }
    ],
    experiment_name="test",
    objective_name="evaluation",
    evaluation_function=eval_s,
    total_trials=50, # Optional.
)

pickle.dump([best_parameters, values, experiment, model], open("saved_optim"), protocol=None, fix_imports=True)
