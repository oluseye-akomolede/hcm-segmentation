"""Source-inspecting tests must not depend on the working directory.

Regression target: running pytest from a directory other than the repo root
made ``test_lifecycle::test_no_invalid_finish_status_in_module`` and
``test_trainer_lifecycle::test_module_has_guarded_delete`` fail with
FileNotFoundError, because they opened ``ecs_hcm_model/core/trainer.py`` by
relative path. Any CI with a different rootdir would have gone red for a
reason that had nothing to do with the code under test.
"""

import os

from conftest import (
    repo_path,
    trainer_source,
    _selftest_repo_path_is_cwd_independent,
)


class TestRepoPathIsCwdIndependent:
    def test_resolves_from_a_foreign_cwd(self):
        _selftest_repo_path_is_cwd_independent()

    def test_trainer_source_is_readable_from_anywhere(self):
        here = os.getcwd()
        try:
            os.chdir("/")
            src = trainer_source()
        finally:
            os.chdir(here)
        assert "def train_model_for_epochs" in src

    def test_repo_path_points_inside_this_repo(self):
        p = repo_path("ecs_hcm_model", "core", "trainer.py").resolve()
        assert p.is_file()
        assert "ecs_hcm_model" in p.parts
