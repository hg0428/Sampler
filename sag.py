import re
from sympy import sympify, latex
from sympy.parsing.latex import parse_latex

delimiters = [
    r"\$",
    r"\\\(",
    r"\\\[",
    r"\\\]",
    r"\\begin{equation}",
    r"\\begin{align}",
    r"\\begin{gather}",
]


def is_fully_latex(text):
    # Remove whitespace from the beginning and end of the string
    text = text.strip()

    # If the string is empty, it's not LaTeX
    if not text:
        return False

    # Define patterns for common LaTeX commands and structures
    latex_patterns = [
        r"\\[a-zA-Z]+",  # LaTeX commands
        r"\{.*?\}",  # Curly braces
        r"\[.*?\]",  # Square brackets
        r"\$.*?\$",  # Inline math mode
        r"_[a-zA-Z0-9]",  # Subscript
        r"\^[a-zA-Z0-9]",  # Superscript
    ]

    # Join patterns into a single regex
    latex_regex = "|".join(latex_patterns)

    # Remove all LaTeX-like structures from the string
    cleaned_text = re.sub(latex_regex, "", text)

    # Check if the remaining text contains only numbers, operators, or whitespace
    remaining_chars = set(cleaned_text.strip())
    allowed_chars = set("0123456789+-*/(),= ")

    return len(remaining_chars - allowed_chars) == 0


def evaluate_latex(equation):
    pattern = r"^(.*?)(\s*)=(\s*)$"

    match = re.match(pattern, equation)

    if match:
        # Group 1: Expression to the left of the equals sign
        expression = match.group(1).strip()

        # Group 2: Spaces to the left of the equals sign
        spaces_left = len(match.group(2))

        # Group 3: Spaces to the right of the equals sign
        spaces_right = len(match.group(3))
        try:
            parsed_exp = parse_latex(expression)

            simplified_exp = parsed_exp.simplify()
        except:
            return None

        return simplified_exp, spaces_left, spaces_right


def finish_last_equation(text):
    last_line = text.rsplit("\n", 1)[-1]
    if re.search(rf"{'|'.join(delimiters)}", last_line):
        split = re.split(
            rf"{'|'.join(delimiters)}",
            last_line,
            maxsplit=1,
        )
        equation = split[-1]
        x = evaluate_latex(equation)
        if x:
            simplified_exp, spaces_left, spaces_right = x
            return text + " " * (spaces_left - spaces_right) + latex(simplified_exp)

    if is_fully_latex(last_line):
        x = evaluate_latex(last_line)
        if x:
            simplified_exp, spaces_left, spaces_right = x
            return text + " " * (spaces_left - spaces_right) + latex(simplified_exp)
    match = re.search(r"([\d\.\+\-\*\/\(\)\^,]+)(\s*)=(\s*)$", last_line)
    if match:
        # Group 1: Expression to the left of the equals sign
        expression = match.group(1).strip()

        # Group 2: Spaces to the left of the equals sign
        spaces_left = len(match.group(2))

        # Group 3: Spaces to the right of the equals sign
        spaces_right = len(match.group(3))

        try:
            parsed_exp = parse_latex(expression)

            simplified_exp = parsed_exp.simplify()
        except:
            return None

        return text + " " * (spaces_left - spaces_right) + latex(simplified_exp)


if __name__ == "__main__":
    # Test the function
    test_cases = [
        "\\frac{2}{2} + \\frac{3}{2} =",
        "\\[1+1=",
        "We know that \\[\\frac{7}{5} + \\frac{3}{5} =",
        "Here's a simple math equation: 1+1=",
        "12 * 12 = ",
        "139 \\cdot 11 =  ",
        "139 x 11  = ",
        "This is a text without an equation.",
        """To find the product of 178634 and 17983432, I'll perform the calculation.

178,634 * 17,983,432 =""",
        """To find the product of 178634 and 17983432, we can use the standard multiplication algorithm. However, for large numbers, it's often more practical to use a calculator or a computer for exact results. Here, I'll provide the exact product using a calculator.

First, we input the numbers into a calculator:

\\[ 178634 \\times 17983432 =""",
    ]

    for case in test_cases:
        result = finish_last_equation(case)
        print(f"Extracted equation: {result}")
        print()
