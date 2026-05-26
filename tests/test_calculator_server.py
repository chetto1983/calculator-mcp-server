from __future__ import annotations

from typing import get_type_hints

import calculator_server as calc


def test_calculate_supports_caret_power_notation() -> None:
    assert calc.calculate("(2 + 3)^2") == {"result": 25}


def test_calculate_rejects_attribute_access() -> None:
    result = calc.calculate("sin.__self__.__dict__")

    assert "error" in result
    assert "Unsupported expression element" in result["error"]


def test_symbolic_tools_accept_caret_power_notation() -> None:
    assert calc.solve_equation("x^2 - 5*x + 6 = 0") == {"solutions": "[2, 3]"}
    assert calc.differentiate("x^3") == {"result": "3*x**2"}
    assert calc.integrate("x^2") == {"result": "x**3/3"}
    assert calc.expand("(x + 1)^2") == {"result": "x**2 + 2*x + 1"}
    assert calc.factorize("x^2 + 2*x + 1") == {"result": "(x + 1)**2"}


def test_vector_tools_handle_multi_element_vectors() -> None:
    assert calc.vector_dot_product([1, 2, 3], [4, 5, 6]) == {"result": 32}
    assert calc.vector_cross_product([1, 0, 0], [0, 1, 0]) == {"result": [0, 0, 1]}
    assert calc.vector_magnitude([3, 4]) == {"result": 5.0}


def test_vector_annotations_are_variable_length_arrays() -> None:
    dot_hints = get_type_hints(calc.vector_dot_product)
    cross_hints = get_type_hints(calc.vector_cross_product)
    magnitude_hints = get_type_hints(calc.vector_magnitude)

    assert dot_hints["vector_a"] == list[float]
    assert dot_hints["vector_b"] == list[float]
    assert cross_hints["vector_a"] == list[float]
    assert cross_hints["vector_b"] == list[float]
    assert magnitude_hints["vector"] == list[float]


def test_matrix_determinant_returns_scalar() -> None:
    assert calc.matrix_determinant([[1, 2], [3, 4]]) == {"result": -2.0}


def test_plot_function_is_headless_and_returns_png_metadata() -> None:
    result = calc.plot_function("x^2", start=-2, end=2, step=5)

    assert result["result"] == "Plot generated successfully."
    assert result["format"] == "png"
    assert result["points"] == 5
    assert result["bytes"] > 0


def test_plot_function_rejects_non_positive_sample_count() -> None:
    assert calc.plot_function("x^2", step=0) == {"error": "step must be positive"}
