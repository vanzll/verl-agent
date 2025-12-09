from typing import List, Tuple
import re
import numpy as np
from agent_system.environments.env_package.math.utils import extract_answer

def math_projection(actions: List[str], check_think_tag: bool = False) -> Tuple[List[str], List[int]]:
    """
    Project a list of LLM actions into (results, valids).

    Criteria:
      1) Contains <think>...</think>;
      2) The content can be successfully extracted from the last occurrence of \\boxed{...} (according to the given logic);
    Returns:
      actions
      valids
    """
    valids: List[int] = []

    for original_str in actions:

        # 1) Check for <think>...</think>
        if check_think_tag:
            has_think = re.search(r"<think>(.*?)</think>", original_str, re.DOTALL | re.IGNORECASE) is not None
        else:
            has_think = True

        # 2) Extract using the provided boxed logic
        boxed_inner = extract_answer(original_str)

        is_valid = int(has_think and (boxed_inner is not None))
        valids.append(is_valid)

    assert len(actions) == len(valids)
    valids = np.array(valids, dtype=bool)

    return actions, valids