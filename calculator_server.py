from __future__ import annotations

import argparse
import ast
import math
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mcp.server.fastmcp import FastMCP
from scipy import stats
from sympy import diff, solve, symbols, sympify
from sympy import integrate as sympy_integrate

# Create MCP Server
app = FastMCP(
    name="Mathematical Calculator",
    instructions="A server for complex mathematical calculations",
    dependencies=["numpy", "scipy", "sympy", "matplotlib"],
)

TRANSPORT = "stdio"

ALLOW_FUNCTION = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "cot": lambda x: 1 / math.tan(x),
    "csc": lambda x: 1 / math.sin(x),
    "sec": lambda x: 1 / math.cos(x),
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "factorial": math.factorial,
    "gamma": math.gamma,
    "erf": math.erf,
    "erfc": math.erfc,
    "lgamma": math.lgamma,
    "degrees": math.degrees,
    "radians": math.radians,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
    "isqrt": math.isqrt,
    "prod": np.prod,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "cumsum": np.cumsum,
    "cumprod": np.cumprod,
    "clip": np.clip,
    "unique": np.unique,
    "sort": np.sort,
    "argsort": np.argsort,
    "argmax": np.argmax,
}

ALLOWED_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: lambda value: +value,
    ast.USub: lambda value: -value,
}


def normalize_expression(expression: str) -> str:
    """Normalize common math notation before parsing/evaluation."""
    return expression.replace("^", "**")


def to_jsonable(value: Any) -> Any:
    """Convert common numpy/sympy outputs into JSON-serializable values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def safe_eval_expression(expression: str) -> Any:
    """Evaluate a math expression using a small AST whitelist."""
    node = ast.parse(expression, mode="eval")
    return _eval_ast(node)


def _eval_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.List):
        return [_eval_ast(element) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(element) for element in node.elts)
    if isinstance(node, ast.Name):
        if node.id in {"pi", "e"}:
            return ALLOW_FUNCTION[node.id]
        raise NameError(f"name '{node.id}' is not defined")
    if isinstance(node, ast.BinOp):
        operator = ALLOWED_BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operator(_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp):
        operator = ALLOWED_UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operator(_eval_ast(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct calls to approved math functions are allowed")
        function = ALLOW_FUNCTION.get(node.func.id)
        if function is None or not callable(function):
            raise NameError(f"name '{node.func.id}' is not defined")
        args = [_eval_ast(arg) for arg in node.args]
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("Expanded keyword arguments are not allowed")
            kwargs[keyword.arg] = _eval_ast(keyword.value)
        return function(*args, **kwargs)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


@app.tool()
def calculate(expression: str) -> dict:
    """
    Evaluates a mathematical expression and returns the result.

    Supports basic operators (+, -, *, /, **, %, ^), mathematical functions
    (sin, cos, tan, exp, log, log10, sqrt), and constants (pi, e).
    Uses an AST whitelist for safe execution.

    Args:
        expression: The mathematical expression to evaluate as a string.
                   Examples: "2 + 2", "sin(pi/4)", "sqrt(16) * 2", "log(100, 10)"

    Returns:
        On success: {"result": <calculated value>}
        On error: {"error": <error message>}

    Examples:
        >>> calculate("2 * 3 + 4")
        {'result': 10}
        >>> calculate("sin(pi/2)")
        {'result': 1.0}
        >>> calculate("sqrt(16)")
        {'result': 4.0}
        >>> calculate("(2 + 3)^2")
        {'result': 25}

    Notes:
        - Use 'x' as the variable (e.g., x**2, not x^2)
        - Multiplication must be explicitly indicated with * (e.g., 2*x, not 2x)
        - Powers can be represented with ** or ^ (e.g., x**2 or x^2)
    """
    try:
        expression = normalize_expression(expression)
        result = safe_eval_expression(expression)
        return {"result": to_jsonable(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def solve_equation(equation: str) -> dict:
    """
    Solves an algebraic equation for x and returns all solutions.

    The equation must contain exactly one equality sign (=) and use a
    variable x. Can solve polynomial, trigonometric, and other equations
    supported by SymPy.

    Args:
        equation: The equation to solve as a string.
                 Format: '<left side> = <right side>'
                 Examples: "x**2 - 5*x + 6 = 0", "sin(x) = 0.5", "2*x + 3 = 7"

    Returns:
        On success: {"solutions": <list of solutions as string>}
        On error: {"error": <error message>}

    Examples:
        >>> solve_equation("x**2 - 5*x + 6 = 0")
        {'solutions': '[2, 3]'}
        >>> solve_equation("2*x + 3 = 7")
        {'solutions': '[2]'}
        >>> solve_equation("x = 0")
        {'solutions': '[0]'}

    Notes:
        - Use 'x' as the variable (e.g., x**2, not x^2)
        - Multiplication must be explicitly indicated with * (e.g., 2*x, not 2x)
        - Powers can be represented with ** or ^ (e.g., x**2 or x^2)
    """
    try:
        x = symbols("x")
        # Split the equation into left and right sides
        parts = equation.split("=")
        if len(parts) != 2:
            return {"error": "Equation must contain an '=' sign"}

        left = sympify(normalize_expression(parts[0].strip()))
        right = sympify(normalize_expression(parts[1].strip()))

        # Solve the equation
        solutions = solve(left - right, x)
        return {"solutions": str(solutions)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def differentiate(expression: str, variable: str = "x") -> dict:
    """
    Computes the derivative of a mathematical expression with respect to a variable.

    Supports polynomials, trigonometric functions, exponential functions,
    logarithms, and other functions supported by SymPy.

    Args:
        expression: The mathematical expression to differentiate as a string.
                   Examples: "x**2", "sin(x)", "exp(x)", "log(x)"
        variable: The variable with respect to which to differentiate. Default is "x".
                 Optionally, other variables can be specified.

    Returns:
        On success: {"result": <derivative as string>}
        On error: {"error": <error message>}

    Examples:
        >>> differentiate("x**2")
        {'result': '2*x'}
        >>> differentiate("sin(x)")
        {'result': 'cos(x)'}
        >>> differentiate("x*y", "y")
        {'result': 'x'}
        >>> differentiate("exp(x)")
        {'result': 'exp(x)'}

    Notes:
        - Use mathematical notation with explicit operators (* for multiplication)
        - Powers can be represented with ** or ^ (e.g., x**2 or x^2)
        - For trigonometric functions, use sin(x), cos(x), etc.
        - Only support for one variable at a time (implicit differentiation not supported)
    """
    try:
        var = symbols(variable)
        expr = sympify(normalize_expression(expression))
        result = diff(expr, var)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def integrate(expression: str, variable: str = "x") -> dict:
    """
    Computes the indefinite integral of a mathematical expression with respect to a variable.

    Supports polynomials, trigonometric functions, exponential functions,
    logarithms, and other functions supported by SymPy.

    Args:
        expression: The mathematical expression to integrate as a string.
                   Examples: "x**2", "sin(x)", "exp(x)", "1/x"
        variable: The variable with respect to which to integrate. Default is "x".
                 Optionally, other variables can be specified.

    Returns:
        On success: {"result": <integral as string>}
        On error: {"error": <error message>}

    Examples:
        >>> integrate("x**2")
        {'result': 'x**3/3'}
        >>> integrate("sin(x)")
        {'result': '-cos(x)'}
        >>> integrate("exp(x)")
        {'result': 'exp(x)'}
        >>> integrate("1/x")
        {'result': 'log(x)'}
        >>> integrate("x*y", "y")
        {'result': 'x*y**2/2'}

    Notes:
        - The result is the indefinite integral without the constant of integration
        - Complex expressions may be returned in simplified form
    """
    try:
        var = symbols(variable)
        expr = sympify(normalize_expression(expression))
        result = sympy_integrate(expr, var)  # Use sympy_integrate instead of integrate
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def mean(data: list[float]) -> dict:
    """
    Computes the mean of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <mean value>}
        On error: {"error": <error message>}

    Examples:
        >>> mean([1, 2, 3, 4])
        {'result': 2.5}
        >>> mean([10, 20, 30])
        {'result': 20.0}
    """
    try:
        result = float(np.mean(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def variance(data: list[float]) -> dict:
    """
    Computes the variance of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <variance value>}
        On error: {"error": <error message>}

    Examples:
        >>> variance([1, 2, 3, 4])
        {'result': 1.25}
    """
    try:
        result = float(np.var(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def standard_deviation(data: list[float]) -> dict:
    """
    Computes the standard deviation of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <standard deviation value>}
        On error: {"error": <error message>}

    Examples:
        >>> standard_deviation([1, 2, 3, 4])
        {'result': 1.118033988749895}
    """
    try:
        result = float(np.std(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def median(data: list[float]) -> dict:
    """
    Computes the median of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <median value>}
        On error: {"error": <error message>}

    Examples:
        >>> median([1, 2, 3, 4])
        {'result': 2.5}
    """
    try:
        result = float(np.median(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def mode(data: list[float]) -> dict:
    """
    Computes the mode of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <mode value>}
        On error: {"error": <error message>}

    Examples:
        >>> mode([1, 2, 2, 3])
        {'result': 2.0}
        >>> mode([1, 1, 2, 2])
        {'result': 1.0}
        >>> mode([])
        {'error': 'Cannot compute mode of empty array'}
    """
    try:
        if not data:
            return {"error": "Cannot compute mode of empty array"}
        # Adjusted for newer SciPy versions
        mode_result = stats.mode(data, keepdims=False)
        return {"result": float(mode_result.mode)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def correlation_coefficient(data_x: list[float], data_y: list[float]) -> dict:
    """
    Computes the Pearson correlation coefficient between two lists of numbers.

    Args:
        data_x: The first list of numerical values.
        data_y: The second list of numerical values.

    Returns:
        On success: {"result": <correlation coefficient>}
        On error: {"error": <error message>}

    Examples:
        >>> correlation_coefficient([1, 2, 3], [4, 5, 6])
        {'result': 1.0}
    """
    try:
        result = np.corrcoef(data_x, data_y)[0, 1]
        return {"result": float(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def linear_regression(data: list[tuple[float, float]]) -> dict:
    """
    Performs linear regression on a set of points and returns the slope and intercept.

    Args:
        data: A list of tuples, where each tuple contains (x, y) coordinates.

    Returns:
        On success: {"slope": <slope value>, "intercept": <intercept value>}
        On error: {"error": <error message>}

    Examples:
        >>> linear_regression([(1, 2), (2, 3), (3, 5)])
        {'slope': 1.5, 'intercept': 0.3333333333333335}
    """
    try:
        x = np.array([point[0] for point in data])
        y = np.array([point[1] for point in data])
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return {"slope": float(slope), "intercept": float(intercept)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def confidence_interval(data: list[float], confidence: float = 0.95) -> dict:
    """
    Computes the confidence interval for the mean of a dataset.

    Args:
        data: A list of numerical values.
        confidence: The confidence level (default is 0.95).

    Returns:
        On success: {"confidence_interval": <(lower_bound, upper_bound)>}
        On error: {"error": <error message>}

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> confidence_interval([1, 2, 3, 4])  # doctest: +ELLIPSIS
        {'confidence_interval': (0.445739743239..., 4.554260256760...)}
    """
    try:
        mean_value = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        margin_of_error = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return {
            "confidence_interval": (
                float(mean_value - margin_of_error),
                float(mean_value + margin_of_error),
            )
        }
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def matrix_addition(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> dict:
    """
    Adds two matrices.

    Args:
        matrix_a: The first matrix as a list of lists.
        matrix_b: The second matrix as a list of lists.

    Returns:
        On success: {"result": <resulting matrix>}
        On error: {"error": <error message>}

    Examples:
        >>> matrix_addition([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        {'result': [[6, 8], [10, 12]]}
    """
    try:
        result = np.add(matrix_a, matrix_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def matrix_multiplication(
    matrix_a: list[list[float]], matrix_b: list[list[float]]
) -> dict:
    """
    Multiplies two matrices.

    Args:
        matrix_a: The first matrix as a list of lists.
        matrix_b: The second matrix as a list of lists.

    Returns:
        On success: {"result": <resulting matrix>}
        On error: {"error": <error message>}

    Examples:
        >>> matrix_multiplication([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        {'result': [[19, 22], [43, 50]]}
    """
    try:
        result = np.dot(matrix_a, matrix_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def matrix_transpose(matrix: list[list[float]]) -> dict:
    """
    Transposes a matrix.

    Args:
        matrix: The matrix to transpose as a list of lists.

    Returns:
        On success: {"result": <transposed matrix>}
        On error: {"error": <error message>}

    Examples:
        >>> matrix_transpose([[1, 2], [3, 4]])
        {'result': [[1, 3], [2, 4]]}
    """
    try:
        result = np.transpose(matrix).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def matrix_determinant(matrix: list[list[float]]) -> dict:
    """
    Computes the determinant of a square matrix.

    Args:
        matrix: A square matrix as a list of rows.

    Returns:
        On success: {"result": <determinant>}
        On error: {"error": <error message>}

    Examples:
        >>> matrix_determinant([[1, 2], [3, 4]])
        {'result': -2.0}
    """
    try:
        result = np.linalg.det(matrix)
        return {"result": round(float(result), 10)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def vector_dot_product(vector_a: list[float], vector_b: list[float]) -> dict:
    """
    Computes the dot product of two equal-length vectors.

    Args:
        vector_a: The first vector as a list of numbers.
        vector_b: The second vector as a list of numbers.

    Returns:
        On success: {"result": <scalar dot product>}
        On error: {"error": <error message>}

    Examples:
        >>> vector_dot_product([1, 2], [7, 8])
        {'result': 23}
    """
    try:
        result = np.dot(vector_a, vector_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def vector_cross_product(vector_a: list[float], vector_b: list[float]) -> dict:
    """
    Computes the cross product of two 2D or 3D vectors.

    Args:
        vector_a: The first vector as a list of numbers.
        vector_b: The second vector as a list of numbers.

    Returns:
        On success: {"result": <resulting vector>}
        On error: {"error": <error message>}

    Examples:
        >>> vector_cross_product([1, 2, 3], [4, 5, 6])
        {'result': [-3, 6, -3]}
    """
    try:
        result = np.cross(vector_a, vector_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def vector_magnitude(vector: list[float]) -> dict:
    """
    Computes the magnitude of a vector.

    Args:
        vector: A vector as a list of numbers.

    Returns:
        On success: {"result": <scalar magnitude>}
        On error: {"error": <error message>}

    Examples:
        >>> vector_magnitude([1, 2, 3])
        {'result': 3.7416573867739413}
    """
    try:
        result = np.linalg.norm(vector).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def plot_function(
    expression: str, start: int = -10, end: int = 10, step: int = 100
) -> dict:
    """
    Plots a graph of y = f(x).

    Args:
        expression: The function expression as a string.
        start: The first x value to plot.
        end: The final x value to plot.
        step: The number of x samples to render.

    Returns:
        On success: {"result": "Plot generated successfully.", "format": "png", ...}
        On error: {"error": <error message>}

    Notes:
        - Use 'x' as the variable (e.g., x**2, not x^2)
        - Multiplication must be explicitly indicated with * (e.g., 2*x, not 2x)
        - Powers can be represented with ** or ^ (e.g., x**2 or x^2)
    """
    x = sp.Symbol("x")
    try:
        if step <= 0:
            return {"error": "step must be positive"}
        expression = sp.sympify(normalize_expression(expression))
        f = sp.lambdify(x, expression, "numpy")
        x_values = np.linspace(start, end, step)
        y_values = f(x_values)
        fig, ax = plt.subplots()
        # Create quadrant graph
        ax.spines["left"].set_position("center")
        ax.spines["bottom"].set_position("center")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.plot(x_values, y_values)
        ax.set_xlabel("x", loc="right")
        ax.set_ylabel("f(x)", loc="top")
        ax.set_title(f"Graph of ${sp.latex(expression)}$")
        ax.grid(True)
        image = BytesIO()
        fig.savefig(image, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return {
            "result": "Plot generated successfully.",
            "format": "png",
            "points": int(len(x_values)),
            "bytes": len(image.getvalue()),
        }
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def summation(expression: str, start: int = 0, end: int = 10) -> dict:
    """
    Calculates the summation of a function from start to end.

    Args:
        expression: The expression of function x as a string.
        start: The starting value of the summation.
        end: The ending value of the summation.

    Returns:
        On success: {"result": <resulting summation>}
        On error: {"error": <error message>}

    Examples:
        >>> summation("x**2", 0, 10)
        {'result': 385}
    """
    try:
        x = sp.Symbol("x")
        expr = sp.sympify(normalize_expression(expression))
        summation = sp.Sum(expr, (x, start, end))
        result = summation.doit()
        return {"result": int(result) if result.is_integer else float(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def expand(expression: str) -> dict:
    """
    Expands an expression.

    Args:
        expression: The expression to expand as a string.

    Returns:
        On success: {"result": <expanded expression>}
        On error: {"error": <error message>}

    Examples:
        >>> expand("(x + 1)**2")
        {'result': 'x**2 + 2*x + 1'}
    """
    try:
        expanded_expression = sp.expand(sp.sympify(normalize_expression(expression)))
        return {"result": str(expanded_expression)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def factorize(expression: str) -> dict:
    """
    Factorizes an expression.

    Args:
        expression: The expression to factorize as a string.

    Returns:
        On success: {"result": <factored expression>}
        On error: {"error": <error message>}

    Examples:
        >>> factorize("x**2 + 2*x + 1")
        {'result': '(x + 1)**2'}
    """
    try:
        factored_expression = sp.factor(sp.sympify(normalize_expression(expression)))
        return {"result": str(factored_expression)}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Mathematical Calculator MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Use STDIO transport (default)")
    parser.add_argument("--sse", action="store_true", help="Use SSE transport")
    args = parser.parse_args()

    transport = "sse" if args.sse else TRANSPORT
    app.run(transport=transport)


if __name__ == "__main__":
    main()
