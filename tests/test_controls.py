import pytest
from covt_pilot.controls import donor_for, token_derangement


def test_token_shuffle_preserves_values_and_moves_every_slot():
    for seed in range(20):
        p=token_derangement(seed)
        assert sorted(p)==list(range(4))
        assert all(i!=j for i,j in enumerate(p))
        assert p==token_derangement(seed)


def test_donor_never_comes_from_same_family():
    target={'id':'a0','family':'a'}
    candidates=[target,{'id':'a1','family':'a'},{'id':'b0','family':'b'}]
    for seed in range(20):
        assert donor_for(target,candidates,seed)['id']=='b0'
    with pytest.raises(ValueError,match='another family'):
        donor_for(target,candidates[:2],0)
