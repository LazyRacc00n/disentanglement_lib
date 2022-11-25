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

"""Different studies that can be reproduced."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from my_initial_experiment import sweep as my_initial_experiment
from overfittiamo_tutti_insieme import sweep as overfittiamo_tutti_insieme
from only_disentangled import sweep as only_disentangled
from no_shape import sweep as no_shape

STUDIES = {
    "my_initial_experiment": my_initial_experiment.MyInitialExperiment(),
    "overfittiamo_tutti_insieme": overfittiamo_tutti_insieme.OverfittiamoTuttiInsieme(),
    "only_disentangled": only_disentangled.OnlyDisentangled()
    "no_shape": no_shape.NoShape()
}
