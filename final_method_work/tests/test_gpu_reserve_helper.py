from malsnif.utils.gpu_guard import _parse_device


def test_device_string_normalization():
    assert _parse_device("1") == 1
    assert _parse_device("cuda:2") == 2
