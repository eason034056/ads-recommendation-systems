import torch

from adsrec.models import DeepFM, ESMM, SemanticGRU


def test_deepfm_starts_with_unsaturated_logits():
    torch.manual_seed(0)
    logits = DeepFM([1000] * 12).eval()(torch.randint(0, 1000, (256, 12)))
    assert logits.abs().max() < 1


def test_semantic_gru_likelihood_sums_to_one_over_all_codes():
    torch.manual_seed(0)
    model = SemanticGRU(item_count=5, vocabulary_sizes=[2, 3, 2]).eval()
    codes = torch.cartesian_prod(torch.arange(2), torch.arange(3), torch.arange(2))
    with torch.no_grad():
        state = model.encode(torch.tensor([[1, 2, 3]])).expand(len(codes), -1)
        total = model.log_likelihood(state, codes).exp().sum()
    torch.testing.assert_close(total, torch.tensor(1.0))


def test_esmm_ctcvr_is_ctr_times_cvr():
    torch.manual_seed(0)
    pctr, pcvr, pctcvr = ESMM([10] * 3).eval()(torch.randint(0, 10, (8, 3)))
    torch.testing.assert_close(pctcvr, pctr * pcvr)
