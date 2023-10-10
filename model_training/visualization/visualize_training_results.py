from typing import Dict, Optional
from matplotlib import pyplot as plt


def visualize_training_results(results: Dict, training_results_name: Optional[str] = None):
    model_type = results.get('model_type', 'N/NA model type')

    for eval_type, eval_metrics in results['training_results'].items():
        for metric_name, metric_val_list in eval_metrics.items():

            plt.plot(range(len(metric_val_list)), metric_val_list)
            plt.xlabel("Epoch Num")
            plt.ylabel(f"{eval_type} {metric_name}")
            plt.title(f"{eval_type} {metric_name} of model {model_type}")
            plt.xticks(range(0, len(metric_val_list)))

            if training_results_name:
                plt.savefig(f"{training_results_name}_{eval_type}_{metric_name}.png")
            plt.show()





