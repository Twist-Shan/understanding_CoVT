from covt_pilot.diagnostics import parse_choice


def test_choice_parser_does_not_extract_labels_from_reasoning():
    assert parse_choice('<think>B might be near.</think><answer>A.</answer>', ['A','B']) == 'A'
    assert parse_choice('A or B', ['A','B']) is None
    assert parse_choice('The answer is A', ['A','B']) is None
    assert parse_choice('left.', ['left','right']) == 'left'
    assert parse_choice(r'\boxed{B}', ['A','B']) == 'B'
