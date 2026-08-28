import pytest

from advancewars import env, parallel_env, raw_env


pettingzoo_test = pytest.importorskip("pettingzoo.test")


def test_official_pettingzoo_api():
    pettingzoo_test.api_test(raw_env(), num_cycles=25, verbose_progress=False)


def test_official_pettingzoo_api_with_fog():
    pettingzoo_test.api_test(raw_env(fog=True), num_cycles=25, verbose_progress=False)


def test_official_pettingzoo_wrapped_env_api():
    pettingzoo_test.api_test(env(), num_cycles=25, verbose_progress=False)


def test_official_pettingzoo_parallel_api():
    pettingzoo_test.parallel_api_test(parallel_env(), num_cycles=25)
