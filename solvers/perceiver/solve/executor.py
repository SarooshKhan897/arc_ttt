"""Code execution for transform functions."""

import re
from typing import Any

import numpy as np
from scipy import ndimage


def parse_llm_response(response: str) -> dict[str, Any]:
    """Extract code and explanation from LLM response."""
    result = {"explanation": "", "code": None}

    # Try to find Python code block
    code_match = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if code_match:
        code_start = response.find('```python')
        result['explanation'] = response[:code_start].strip()
        result['code'] = code_match.group(1).strip()
    else:
        # Try generic code block with import/def transform
        code_match = re.search(r'```\s*(import.*?def transform.*?)```', response, re.DOTALL)
        if code_match:
            result['code'] = code_match.group(1).strip()

    return result


def execute_transform(code: str, input_grid: np.ndarray) -> np.ndarray:
    """
    Execute transform code on an input grid.

    Args:
        code: Python code containing a `transform(grid)` function
        input_grid: The input grid to transform

    Returns:
        The transformed grid as a numpy array

    Raises:
        ValueError: If no valid code or transform function
        Exception: If code execution fails
    """
    if not code:
        raise ValueError("No code provided")

    # Set up execution namespace with allowed imports
    namespace = {
        'np': np,
        'numpy': np,
        'ndimage': ndimage,
    }

    # Execute the code to define the transform function
    exec(code, namespace)

    if 'transform' not in namespace:
        raise ValueError("No 'transform' function found in code")

    # Call the transform function
    result = namespace['transform'](input_grid.copy())

    return np.array(result)


def test_on_examples(
    code: str,
    train_examples: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """
    Test code on training examples.

    Args:
        code: Python code with transform function
        train_examples: List of {"input": grid, "output": grid} examples

    Returns:
        (all_passed, feedback_messages)
    """
    feedback = []
    all_passed = True

    for idx, example in enumerate(train_examples):
        input_grid = np.array(example['input'])
        expected_output = np.array(example['output'])

        try:
            actual_output = execute_transform(code, input_grid)

            if np.array_equal(actual_output, expected_output):
                feedback.append(f"Example {idx + 1}: ✓ Passed")
            else:
                all_passed = False
                # Generate diff info
                if actual_output.shape != expected_output.shape:
                    feedback.append(
                        f"Example {idx + 1}: ✗ Shape mismatch - "
                        f"expected {expected_output.shape}, got {actual_output.shape}"
                    )
                else:
                    diff_count = np.sum(actual_output != expected_output)
                    total = expected_output.size
                    feedback.append(
                        f"Example {idx + 1}: ✗ {diff_count}/{total} cells wrong"
                    )
        except Exception as e:
            all_passed = False
            feedback.append(f"Example {idx + 1}: ✗ Error - {str(e)[:100]}")

    return all_passed, feedback

