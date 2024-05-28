#!python
# -*- coding: utf-8 -*-
import subprocess


def test(fake_process):
    fake_process.allow_unregistered(True)
    process = subprocess.Popen(
        """
        python lca/overall_driver.py \
            --ga_config examples/default/config.ini \
            --verifier_gt examples/default/verifier_probs.json \
            --request examples/default/request_example.json \
            --db_result examples/default/result.json
        """,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
        universal_newlines=True,
    )
    out, errs = process.communicate(timeout=60)

    assert process.returncode == 0
    assert errs is None


if __name__ == '__main__':
    test()
