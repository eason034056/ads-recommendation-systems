import torch

from adsrec.models import DeepFM, ESMM


def test_deepfm_starts_with_unsaturated_logits():
    torch.manual_seed(0)
    logits = DeepFM([1000] * 12).eval()(torch.randint(0, 1000, (256, 12)))
    assert logits.abs().max() < 1


def test_esmm_ctcvr_is_ctr_times_cvr():
    torch.manual_seed(0)
    pctr, pcvr, pctcvr = ESMM([10] * 3).eval()(torch.randint(0, 10, (8, 3)))
    torch.testing.assert_close(pctcvr, pctr * pcvr)
