

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib 
import os
from absl import app
from absl import flags
from absl import logging
import gin 
gin.enter_interactive_mode()
gin.clear_config()
import numpy as np



from config.only_disentangled import sweep as only_disentangled
from postprocessing import postprocess
from evaluation import evaluate



FLAGS = flags.FLAGS
flags.DEFINE_string("study", "only_disentangled",
                    "Name of the study.")
flags.DEFINE_string("output_directory", None,
                    "Output directory of experiments ('{model_num}' will be"
                    " replaced with the model index  and '{study}' will be"
                    " replaced with the study name if present).")
# Model flags. If the model_dir flag is set, then that directory is used and
# training is skipped.
flags.DEFINE_string("model_dir", None, "Directory to take trained model from.")
# Otherwise, the model is trained using the 'model_num'-th config in the study.
flags.DEFINE_integer("model_num", 0,
                     "Integer with model number to train.")
flags.DEFINE_boolean("only_print", False,
                     "Whether to only print the hyperparameter settings.")
flags.DEFINE_boolean("overwrite", True,
                     "Whether to overwrite output directory.")

def main(unused_argv):

  
  # Obtain the study to reproduce.
  study = only_disentangled.OnlyDisentangled()

  # Print the hyperparameter settings.
  if FLAGS.model_dir is None:
    study.print_model_config(FLAGS.model_num)
  else:
    print("Model directory (skipped training):")
    print("--")
    print(FLAGS.model_dir)
  print()
  study.print_postprocess_config()
  print()
  study.print_eval_config()
  if FLAGS.only_print:
    return

  # Set correct output directory.
  if FLAGS.output_directory is None:
    if FLAGS.model_dir is None:
      output_directory = os.path.join("output", "{study}", "{model_num}")
    else:
      output_directory = "output"
  else:
    output_directory = FLAGS.output_directory

  # Insert model number and study name into path if necessary.
  output_directory = output_directory.format(model_num=str(FLAGS.model_num),
                                             study=str(FLAGS.study))

  # Model training (if model directory is not provided).
  if FLAGS.model_dir is None:
    model_dir = os.path.join(output_directory, "model")
  else:
    logging.info("Skipped training...")
    model_dir = FLAGS.model_dir

  # We fix the random seed for the postprocessing and evaluation steps (each
  # config gets a different but reproducible seed derived from a master seed of
  # 0). The model seed was set via the gin bindings and configs of the study.
  random_state = np.random.RandomState(0)

  # We extract the different representations and save them to disk.
  postprocess_config_files = sorted(study.get_postprocess_config_files())
  for config in postprocess_config_files:
    post_name = os.path.basename(config).replace(".gin", "")
    logging.info("Extracting representation %s...", post_name)
    post_dir = os.path.join(output_directory, "postprocessed", post_name)
    postprocess_bindings = [
        "postprocess.random_seed = {}".format(random_state.randint(2**32)),
        "postprocess.name = '{}'".format(post_name)
    ]
    postprocess.postprocess_with_gin(model_dir, post_dir, FLAGS.overwrite,
                                     [config], postprocess_bindings)
  
  # We extract the different representations and save them to disk.
  postprocess_config_files = sorted(study.get_postprocess_config_files())

  # Iterate through the disentanglement metrics.
  eval_configs = sorted(study.get_eval_config_files())
 
  for config in postprocess_config_files:
    post_name = os.path.basename(config).replace(".gin", "")
    post_dir = os.path.join(output_directory, "postprocessed",
                            post_name)

    # Now, we compute all the specified scores.
    for gin_eval_config in eval_configs:
      metric_name = os.path.basename(gin_eval_config).replace(".gin", "")
      logging.info("Computing metric '%s' on '%s'...", metric_name, post_name)
      metric_dir = os.path.join(output_directory, "metrics", post_name,
                                metric_name)
      eval_bindings = [
          "evaluation.random_seed = {}".format(random_state.randint(2**32)),
          "evaluation.name = '{}'".format(metric_name)
      ]
      
      evaluate.evaluate_with_gin(post_dir, metric_dir, FLAGS.overwrite, [gin_eval_config], eval_bindings)




if __name__ == "__main__":
  app.run(main)

