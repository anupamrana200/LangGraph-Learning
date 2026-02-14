from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

load_dotenv()
model = ChatOpenAI()

class chatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


def chatNode(state: chatState):
  messages = state['messages']

  response = model.invoke(messages)

  return {'messages':[response]}


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkPointer = SqliteSaver(conn=conn)

graph = StateGraph(chatState)

graph.add_node('chatNode', chatNode)

graph.add_edge(START, 'chatNode')
graph.add_edge('chatNode', END)

chatbot = graph.compile(checkpointer=checkPointer)


def retrive_all_threads():
  all_threads = set()
  for checkpoint in checkPointer.list(None):
    all_threads.add(checkpoint.config['configurable']['thread_id'])

  return list(all_threads)