from  langchain.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults


@tool()
def divide(a: float, b: float) -> float:
    """Divides two numbers. Returns 'Error: Division by zero' if b=0."""
    try:
        return a / b
    except ZeroDivisionError:
        return "Error: Division by zero"

@tool
def search() -> str:
    """Search for news. Returns top 3 summaries."""
    results = DuckDuckGoSearchResults()
    return "\n".join([f"- {result['title']}" for result in results[:3]])
tools = [divide, search]