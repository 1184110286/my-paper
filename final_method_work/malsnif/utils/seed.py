import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    deterministic = os.environ.get("MALSNIF_DETERMINISTIC", "0").lower() in {"1", "true", "yes", "y", "on"}
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some CUDA deterministic algorithms require this environment variable;
        # setting a default here keeps rigorous experiment scripts self-contained.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    try:
        torch.set_num_threads(int(os.environ.get("MALSNIF_TORCH_THREADS", "1")))
        torch.set_num_interop_threads(int(os.environ.get("MALSNIF_TORCH_INTEROP_THREADS", "1")))
    except Exception:
        pass
