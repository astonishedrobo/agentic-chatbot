from dataclasses import dataclass
from typing import List
import operator
from typing_extensions import Annotated
import uuid
import asyncio

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage, trim_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import get_buffer_string
from langchain_core.messages.utils import count_tokens_approximately

from mem0 import Memory

from utils.tools import *

@dataclass
class GraphState:
    messages: Annotated[List[AnyMessage], operator.add]

class AgenticChatBot:
    def __init__(self, user_id: str, lt_mem: Memory, config: dict):
        self._user_id = user_id
        self._lt_mem = lt_mem
        try:
            self._system_prompt = config["system_prompt"]
        except KeyError as e:
            raise e
        try:
            self._llm_config = config["llm"]
            llm, temperature = self._llm_config.get("model", "openai:gpt-4o"), self._llm_config.get("temperature", 0.7)
            self._llm = init_chat_model(llm, temperature=temperature)
        except KeyError as e:
            raise e
        
        self._thread_id = str(uuid.uuid4())
        self._config = {"configurable": {"thread_id": self._thread_id}, "recursion_limit": config.get("recursion_limit", 25)}
        self._memory = MemorySaver()
        self._setup_tools()
        self._setup_graph()
        

    def _setup_tools(self):
        self._tools = [add_update_memory, fetch_memory]
        self._tools_map = {tool.name: tool for tool in self._tools}

        self._llm_with_tools = self._llm.bind_tools(self._tools)

    def _setup_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("agent", self._executor)
        graph.add_node("execute_tools", self._execute_tools)

        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._tool_exists, {True: "execute_tools", False: END})
        graph.add_edge("execute_tools", "agent")

        self._graph = graph.compile(checkpointer=self._memory)

    def _executor(self, state: GraphState):
        max_context = self._llm_config.get("max_context", None)
        if max_context and self._count_tokens(state.messages) > max_context:
            self._trim_messages(state)

        response = self._llm_with_tools.invoke(state.messages)
        return {"messages": [response]}

    async def _execute_tools(self, state: GraphState):
        message: AIMessage = state.messages[-1]
        tool_calls = getattr(message, "tool_calls", [])

        async def invoke_tool(tool_call):
            tool_fn = self._tools_map.get(tool_call["name"])
            if not tool_fn:
                return ToolMessage(tool_call_id=tool_call["id"], content="Tool doesn't exist!")
            try:
                args = tool_call.get("args", {})
                args.update({"user_id": self._user_id})
                res = await tool_fn.ainvoke(args)
                return ToolMessage(tool_call_id=tool_call["id"], content=str(res))
            except Exception as e:
                return ToolMessage(tool_call_id=tool_call["id"], content=f"Error: {e}")

        tool_messages = await asyncio.gather(*[invoke_tool(tc) for tc in tool_calls])
        return {"messages": tool_messages}

    def _tool_exists(self, state: GraphState):
        message = state.messages[-1]
        return isinstance(message, AIMessage) and bool(getattr(message, "tool_calls", []))

    def _count_tokens(self, messages: List[AnyMessage]) -> int:
        return self._llm.get_num_tokens(get_buffer_string(messages))
    
    def _trim_messages(self, state: GraphState):
        state.messages = trim_messages(
            state.messages,
            token_counter=count_tokens_approximately,
            max_tokens=self._llm_config.get("max_context", 28000),
            strategy="last",
            start_on="human",
            end_on=("human", "assistant", "tool"),
            include_system=True,
            allow_partial=False,
        )
    
    #### Public APIs ####
    def clear_memory(self):
        """
        Clears the long-term memory.
        """
        self._lt_mem.delete_all(user_id=self._user_id)

    def start_new_conversation(self):
        """
        Resets the conversation by creating a new thread ID.
        """
        self._thread_id = str(uuid.uuid4())
        self._config = {"configurable": {"thread_id": self._thread_id}}

    async def run(self, query: str):
        """
        Runs a turn of the conversation with the user's query and returns the output.
        """
        thread_state = self._memory.get(self._config)
        if not thread_state:
            input_data = {
                "messages": [SystemMessage(self._system_prompt), HumanMessage(content=query)],
                "collected_insights": [], "variables": {}
            }
        else:
            input_data = {"messages": [HumanMessage(content=query)]}

        response = await self._graph.ainvoke(input_data, config=self._config)
        return response
