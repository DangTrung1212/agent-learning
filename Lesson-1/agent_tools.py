from  langchain.tools import tool

@tool()
def divide(a: float, b: float) -> float:
    """Divides two numbers. Returns 'Error: Division by zero' if b=0."""
    try:
        return a / b
    except ZeroDivisionError:
        return "Error: Division by zero"

tools = [divide]