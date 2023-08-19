- By default, LLMs are stateless, which means they process each incoming query in isolation, without considering previous interactions.

# ConversationBufferMemory

- This memory implementation stores the entire conversation history as a single string
- Can be less efficient as the conversation grows longer and may lead to excessive repetition if the conversation history is too long for the model's token limit.
- If the token limit of the model is surpassed, the buffer gets truncated to fit within the model's token limit
  - This means that older interactions may be removed from the buffer to accommodate newer ones

```py
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

# TODO: Set your OPENAI API credentials in environemnt variables.
llm = OpenAI(model_name="text-davinci-003", temperature=0)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
conversation.predict(input="Hello!")

```

# ConversationBufferWindowMemory

- This class limits memory size by keeping a list of the most recent K interactions
- maintains a sliding window of these recent interactions, ensuring that the buffer does not grow too large.
- More efficient than ConversationBufferMemory
- downside of using this approach is that it does not maintain the complete conversation history

```py

from langchain.memory import ConversationBufferWindowMemory
from langchain import OpenAI, LLMChain, PromptTemplate

template = """You are ArtVenture, a cutting-edge virtual tour guide for
 an art gallery that showcases masterpieces from alternate dimensions and
 timelines. Your advanced AI capabilities allow you to perceive and understand
 the intricacies of each artwork, as well as their origins and significance in
 their respective dimensions. As visitors embark on their journey with you
 through the gallery, you weave enthralling tales about the alternate histories
 and cultures that gave birth to these otherworldly creations.

{chat_history}
Visitor: {visitor_input}
Tour Guide:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "visitor_input"],
    template=template
)

chat_history=""

convo_buffer_win = ConversationChain(
    llm=llm,
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)
)

```

- The value of k (in this case, 3) represents the number of past messages to be stored in the buffer

# ConversationSummaryMemory

- ConversationSummaryBufferMemory is a memory management strategy that combines the ideas of keeping a buffer of recent interactions in memory and compiling old interactions into a summary
- uses token length rather than number of interactions to determine when to flush interactions

```py
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

# Create a ConversationChain with ConversationSummaryMemory
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm),
    verbose=True
)

# Example conversation
response = conversation_with_summary.predict(input="Hi, what's up?")
print(response)

```

- Using the predict method to have a conversation with the AI, which uses ConversationSummaryBufferMemory to store the conversation's summary and buffer.

```py
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\nCurrent conversation:\n{topic}",
)

from langchain.llms import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(temperature=0)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40),
    verbose=True
)
conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="Just working on writing some documentation!")
response = conversation_with_summary.predict(input="For LangChain! Have you heard of it?")
print(response)

#View documentation to see how it responds in real time https://python.langchain.com/docs/modules/memory/types/summary_buffer

```
