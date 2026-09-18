import numpy as np
from covt_pilot.shift_scenes import generate_shift
from covt_pilot.metrics import raw_difference


def test_shift_geometry_labels_and_selection(tmp_path):
    rows=generate_shift(tmp_path/'shift',families=10,seed=1901)
    assert len(rows)==40
    assert len({r['stratum'] for r in rows})==5
    for family in {r['family'] for r in rows}:
        group={r['variant']:r for r in rows if r['family']==family}
        assert group['base']['label']!=group['depth_swap']['label']
        assert group['base']['label']!=group['label_swap']['label']
        assert group['base']['label']==group['nuisance']['label']
    for row in rows:
        with np.load(tmp_path/'shift'/row['geometry']) as z:
            assert (raw_difference(z['depth'],row)>0)==(row['label']=='A')
