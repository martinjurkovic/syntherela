import os
from shutil import rmtree

import pytest


@pytest.fixture(scope='session', autouse=True)
def cleanup_tmp_dirs():
    """Remove tests/tmp once after all tests in the session have finished."""
    yield
    if os.path.exists('tests/tmp'):
        rmtree('tests/tmp')
