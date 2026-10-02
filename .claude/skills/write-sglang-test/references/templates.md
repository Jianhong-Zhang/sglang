# Test templates

Copy and adapt. Unit-test template and server fixtures are in SKILL.md.

## Contents

- E2E test (small model, server needed)
- E2E test (8B model, server needed, performance)
- Eval accuracy mixins

## E2E test (small model, server needed)

```python
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, suite="base-b-test-1-gpu-small")


class TestMyFeature(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--arg1", "value1"],  # feature-specific args
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_basic_functionality(self):
        response = requests.post(
            self.base_url + "/generate",
            json={"text": "Hello", "sampling_params": {"max_new_tokens": 32}},
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=3)
```

## E2E test (8B model, server needed, performance)

```python
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="base-b-test-1-gpu-large")


class TestMyFeaturePerf(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_latency(self):
        start = time.perf_counter()
        response = requests.post(
            self.base_url + "/generate",
            json={"text": "Hello", "sampling_params": {"max_new_tokens": 128}},
        )
        elapsed = time.perf_counter() - start
        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 5.0, "Latency exceeded threshold")


if __name__ == "__main__":
    unittest.main(verbosity=3)
```

## Eval accuracy mixins

**Design philosophy**: Most test files don't care about eval logic — they only need a "does this feature break model output quality?" sanity check. The mixin pattern separates **what to test** (threshold) from **how to test** (run_eval, assertions, CI summary). Test classes declare thresholds as class attributes; the mixin provides the `test_*` method. Override when you need extra assertions (e.g. EAGLE accept length).

Available mixins in `python/sglang/test/kits/eval_accuracy_kit.py`: `MMLUMixin`, `HumanEvalMixin`, `MGSMEnMixin`, `GSM8KMixin`. Can be combined freely. Read the source for attrs and defaults.

```python
class TestMyFeature(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32
    # test_mmlu is inherited — no code needed
```
