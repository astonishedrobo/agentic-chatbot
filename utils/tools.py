from typing_extensions import Annotated
import os
import yaml

from langchain_core.tools import tool, InjectedToolArg
from mem0 import Memory

lt_mem = None
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

def getlt_mem():
    """Get or create Mem0 client instance"""
    global lt_mem
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = {
        "vector_store": {
            "provider": config["vector_store"]["provider"],
            "config": {
                "host": config["vector_store"]["config"]["host"],
                "port": config["vector_store"]["config"]["ports"][0],
            }
        },
        "graph_store": {
            "provider": config["graph_store"]["provider"],
            "config": {
                "url": config["graph_store"]["config"]["url"],
                "username": config["graph_store"]["config"]["username"],
                "password": config["graph_store"]["config"]["password"]
            }
        },
    }
    if lt_mem is None:
        lt_mem = Memory.from_config(config)
    return lt_mem

@tool
async def add_update_memory(memory: str, user_id: Annotated[str, InjectedToolArg]):
    """
    Add memory to the long-term memory (Automatically updates the memory if there is any conflict).

    Args: 
        memory: The memory to be added or updated. A brief description of the memory or memories.
    """
    lt_mem = getlt_mem()
    _ = lt_mem.add(
        messages=[{"role": "assistant", "content": f"Memory: {memory}"}],
        user_id=user_id,
    )
    
    return "Memory has been added successfully."

@tool
async def fetch_memory(query: str, user_id: Annotated[str, InjectedToolArg]):
    """
    Fetch the long-term memory.

    Args:
        query: The query to filter the fetched memory.

    Returns:
        The fetched memory based on the query for the specified user ID.
    """
    lt_mem = getlt_mem()
    memories = lt_mem.search(query, user_id=user_id)

    return memories