# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hyperparameter sweeps and configs for the study "unsupervised_study_v1".

Challenging Common Assumptions in the Unsupervised Learning of Disentangled
Representations. Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch,
Sylvain Gelly, Bernhard Schoelkopf, Olivier Bachem. arXiv preprint, 2018.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.config import study
import disentanglement_lib.utils.hyperparams as h
from six.moves import range
import os

def get_datasets():
  """Returns all the data sets."""
  return h.sweep(
      "my_dataset.name",
      h.categorical([
          "dsprites_heartonly"
      ]))

def get_visualize_dataset():
  """Returns all the data sets."""
  return h.sweep(
      "dataset.name",
      h.categorical([
          "dsprites_heartonly"
      ]))

def get_num_latent(num):
  return h.fixed("encoder.num_latent", num)


def get_seeds(num):
  """Returns random seeds."""
  return h.sweep("model.random_seed", h.categorical(list(range(num))))


def get_default_models():
  """Our default set of models (6 model * 6 hyperparameters=36 models)."""
  # BetaVAE config.
  model_name = h.fixed("model.name", "beta_vae")
  model_fn = h.fixed("model.model", "@vae()")
  model_steps = h.fixed("model.training_steps", 300000)
  betas = h.fixed("vae.beta", 16.)
  config_beta_vae = h.zipit([model_name, betas, model_fn, model_steps ])

  all_models = h.chainit([
      config_beta_vae
  ])
  return all_models


def get_config():
  """Returns the hyperparameter configs for different experiments."""
  arch_enc = h.fixed("encoder.encoder_fn", "@conv_encoder", length=1)
  arch_dec = h.fixed("decoder.decoder_fn", "@deconv_decoder", length=1)
  architecture = h.zipit([arch_enc, arch_dec])
  return h.product([
      get_datasets(),
      get_visualize_dataset(),
      architecture,
      get_default_models(),
      get_seeds(5),
      get_num_latent(4)
  ])


class NoShape(study.Study):
  """Defines the study for the paper."""

  def get_model_config(self, model_num=0):
    """Returns model bindings and config file."""
    config = get_config()[model_num]
    model_bindings = h.to_bindings(config)
    model_config_file = "config/no_shape/model_configs/shared.gin"
    return model_bindings, model_config_file

  def get_postprocess_config_files(self):
    """Returns postprocessing config files."""
    return ["config/no_shape/postprocess_configs/mean.gin", "config/no_shape/postprocess_configs/sampled.gin", "config/no_shape/postprocess_configs/mean_std.gin"]

  def get_eval_config_files(self):
    """Returns evaluation config files."""
    #files = ["beta_vae_sklearn.gin", "dci.gin", "factor_vae_metric.gin", "mig.gin", "modularity_explicitness.gin", "sap_score.gin", "unsupervised.gin"]
    files=[]
    return [os.path.join(os.getcwd(), "config/no_shape/metric_configs", f) for f in files ]
