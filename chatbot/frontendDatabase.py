import streamlit as st
from backendDatabase import chatbot, retrive_all_threads
from langchain_core.messages import HumanMessage
import uuid #it is used to generate new thread_id

#************************************* Utility Function  ***************************************
def generate_thread_id():
  thread_id = uuid.uuid4()
  return thread_id


def reset_chat():
  thread_id = generate_thread_id()
  st.session_state['thread_id'] = thread_id
  add_thread(st.session_state['thread_id'])
  st.session_state['message_history'] = []

def add_thread(thread_id):
  if thread_id not in st.session_state['chat_threads']:
    st.session_state['chat_threads'].insert(0,thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get("messages", []) if state else []

#************************************** Session Setup *******************************************
if 'message_history' not in st.session_state:
  st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
  st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
  st.session_state['chat_threads'] = retrive_all_threads()

add_thread(st.session_state['thread_id'])

#************************************** Sidebar *************************************************
st.sidebar.title('LangGraph Chatbot')
new_chat_button = st.sidebar.button('New Chat')
if new_chat_button:
  reset_chat()
  st.rerun()

st.sidebar.header('My Conversation')

for thread_id in st.session_state['chat_threads']:
  if st.sidebar.button(str(thread_id)):
    st.session_state['thread_id'] = thread_id
    messages = load_conversation(thread_id)

    temp_messages = []

    for msg in messages:
      if isinstance(msg, HumanMessage):
        role = 'user'
      else: 
        role = 'assistant'
      temp_messages.append({'role': role, 'content':msg.content})
    
    st.session_state['message_history'] = temp_messages




#*************************************** Main UI ************************************************
#Loading the conversation history.
for message in st.session_state['message_history']:
  with st.chat_message(message['role']):
    st.text(message['content'])

#Input from the user.
user_input = st.chat_input("Your question type here...")


if user_input:
  #First add the message to the message history
  st.session_state['message_history'].append({'role':'user', 'content':user_input})
  with st.chat_message('user'):
    st.text(user_input)

  
  Config = {'configurable': {'thread_id':st.session_state['thread_id']}}

  with st.chat_message('assistant'):
    ai_message = st.write_stream(
      message_chunk.content for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=Config,
        stream_mode='messages'
      )
    )
    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})

   