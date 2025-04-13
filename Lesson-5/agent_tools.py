from  langchain.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain.agents import tool


@tool
def search(query: str) -> str:
    """Search for news. then return """
    try:
        search = DuckDuckGoSearchResults()
        results = search.invoke(query)
        print(results)
        return results
    except Exception as e:
        return f"An error occurred: {str(e)}"
tools = [search]
