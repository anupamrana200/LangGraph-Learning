from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()

class chatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


def chatNode(state: chatState):
  messages = state['messages']

  response = model.invoke(messages)

  return {'messages':[response]}

checkPointer = MemorySaver()

graph = StateGraph(chatState)

graph.add_node('chatNode', chatNode)

graph.add_edge(START, 'chatNode')
graph.add_edge('chatNode', END)

chatbot = graph.compile(checkpointer=checkPointer)
