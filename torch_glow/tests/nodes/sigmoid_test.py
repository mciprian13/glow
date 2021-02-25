# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleSigmoidModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleSigmoidModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        if self.inplace:
            other = tensor + tensor
            return other.sigmoid_()
        else:
            other = tensor + tensor
            return other.sigmoid()


class TestSigmoid(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleSigmoidModel(), torch.randn(6)),
            lambda: ("inplace", SimpleSigmoidModel(inplace=True), torch.randn(6)),
        ]
    )
    def test_sigmoid(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::sigmoid"})
