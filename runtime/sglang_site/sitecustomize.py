"""Install KVScope in every process that SGLang spawns.

SGLang starts its scheduler in a new process, so code that runs in the launching script never reaches the
scheduler. Python imports a module named ``sitecustomize`` at start-up in each process that has its directory on
``PYTHONPATH``. Put this directory first on ``PYTHONPATH`` (``eval/common/sglang_engine.py`` does) and set
``EKV_ENABLE=1`` to install what ``runtime.sglang_hooks.register`` is asked for in each of those processes.
"""

import os

if os.environ.get("EKV_ENABLE"):
    from runtime.sglang_hooks import register

    register()
