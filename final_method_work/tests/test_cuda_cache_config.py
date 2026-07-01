from malsnif.config import Config
from malsnif.utils.cuda_memory import empty_cuda_cache


def test_cuda_empty_cache_config_defaults():
    cfg = Config()
    assert cfg.cuda_empty_cache_interval == 0
    assert cfg.cuda_empty_cache_after_epoch is False
    assert cfg.cuda_empty_cache_after_eval is False


def test_empty_cuda_cache_cpu_is_noop():
    assert empty_cuda_cache('cpu') is False
