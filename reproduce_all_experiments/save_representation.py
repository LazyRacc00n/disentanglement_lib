
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
import tensorflow as tf
import time

from disentanglement_lib.config import reproduce

from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.utils import results
from disentanglement_lib.data.ground_truth import named_data
import tensorflow_hub as hub

FLAGS = flags.FLAGS
flags.DEFINE_string("study", "unsupervised_study_v1",
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
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")



def prepare_gin(model_num, model_dir,
                      output_dir,
                      overwrite=False,
                      gin_config_files=None,
                      gin_bindings=None):

  if gin_config_files is None:
    gin_config_files = []
  if gin_bindings is None:
    gin_bindings = []
  
  #gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
  prepare_arguments(model_dir, output_dir, overwrite, model_num=model_num)
  gin.clear_config()


@gin.configurable(
    "evaluation", blacklist=["model_dir", "output_dir", "overwrite"])
def prepare_arguments(model_dir,
             output_dir,
             overwrite=False,
             evaluation_fn=None,
             random_seed=0,
             name="",
             model_num=0):

   # Delete the output directory if it already exists.
  if tf.gfile.IsDirectory(output_dir):
    if overwrite:
      tf.gfile.DeleteRecursively(output_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Set up time to keep track of elapsed time in results.
  experiment_timer = time.time()

  # Automatically set the proper data set if necessary. We replace the active
  # gin config as this will lead to a valid gin config file where the data set
  # is present.
  
  # Obtain the dataset name from the gin config of the previous step.
  gin_config_file = os.path.join(model_dir, "results", "gin",
                                   "postprocess.gin")
  print(gin_config_file)
  gin_dict = results.gin_dict(gin_config_file)
  print(gin_dict)
  with gin.unlock_config():
      gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
          "'", ""))
  dataset = named_data.get_named_ground_truth_data()
  
  # Path to TFHub module of previously trained representation.
  module_path = os.path.join(model_dir, "tfhub")
  with hub.eval_function_for_module(module_path) as f:

    def _representation_function(x):
      """Computes representation vector for input images."""
      output = f(dict(images=x), signature="representation", as_dict=True)
      return np.array(output["default"])
  
    numpy_data, numpy_factors = utils.generate_batch_factor_code(
      dataset, _representation_function, 30000,
      np.random.RandomState(random_seed), 1)
    
    print(os.path.join(model_dir, "train_representation_{}.npy".format(model_num)))
    with open(os.path.join(model_dir, "train_representation_{}.npy".format(model_num)), 'wb') as f:
      np.save(f, numpy_data)
    with open(os.path.join(model_dir, "train_factors_{}.npy".format(model_num)), 'wb') as f:
      np.save(f, numpy_factors)

def main(unused_argv):

  
  # Obtain the study to reproduce.
  study = reproduce.STUDIES[FLAGS.study]

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
  
  logging.info("Skipped training...")
  model_dir = FLAGS.model_dir

  # We fix the random seed for the postprocessing and evaluation steps (each
  # config gets a different but reproducible seed derived from a master seed of
  # 0). The model seed was set via the gin bindings and configs of the study.
  random_state = np.random.RandomState(0)
  
  # We extract the different representations and save them to disk.
  postprocess_config_files = sorted(study.get_postprocess_config_files())

  # Iterate through the disentanglement metrics.
  eval_configs = sorted(study.get_eval_config_files())
 
  for config in postprocess_config_files:
    post_name = os.path.basename(config).replace(".gin", "")
    post_dir = os.path.join(output_directory, "postprocessed",
                            post_name)

    # Now, we compute all the specified scores.
    metric_name = "representation"
    metric_dir = os.path.join(output_directory, metric_name)
    eval_bindings = [
          "evaluation.random_seed = {}".format(random_state.randint(2**32)),
          "evaluation.name = '{}'".format(metric_name)
      ]
     
    gin_eval_config = [
          "my_dataset.name = auto"
      ]

    prepare_gin(FLAGS.model_num, post_dir, metric_dir, FLAGS.overwrite, gin_eval_config, eval_bindings)



if __name__ == "__main__":
  app.run(main)
