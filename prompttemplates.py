from langchain import LLMChain, PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI


import os

os.environ['OPENAI_API_KEY'] = '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'

llm = OpenAI(model_name= "text-davinci-003", temperature=0)

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided, answer
with "I don't know".
Context: Quantum computing is an emerging field that leverages quantum mechanics to solve complex problems faster than classical computers.
...
Question: {query} 
Answer: """

# {Query is now a single dynamic input for the user}

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
) # instantiate PromptTemplate into a variable with the arguments for what the input variable will be and the template we'll be using, there could be multiple templates we could specify

# Create the LLMChain for the prompt
chain = LLMChain(llm=llm, prompt=prompt_template) # chain wraps everything together so you don't have to call things individually

# Set the query you want to ask
input_data = {"query": "What is the main advantage of quantum computing over classical computing?"} 

# Run the LLMChain to get the AI-generated answer
response = chain.run(input_data)

print("Question:", input_data["query"])
print("Answer:", response) # Question and answers are simply there to distinguish between what is the question and is the answer, nothing fancy just strings placed there

##############################################################

# Few Shot Templates

# Use this to include a subset of exaples that will allow the LLm to generate better more precise answers as it will learn off the examples rather than just learning off nothing

llm = OpenAI(model_name="text-davinci-003", temperature=0)

examples = [
    {"animal": "lion", "habitat": "savanna"},
    {"animal": "polar bear", "habitat": "Arctic ice"},
    {"animal": "elephant", "habitat": "African grasslands"}
] # Feed it examples

example_template = """
Animal: {animal}
Habitat: {habitat}
""" # Give it a template 

example_prompt = PromptTemplate(
    input_variables=["animal", "habitat"],
    template=example_template
)

dynamic_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Identify the habitat of the given animal",
    suffix="Animal: {input}\nHabitat:",
    input_variables=["input"],
    example_separator="\n\n",
)

# Create the LLMChain for the dynamic_prompt
chain = LLMChain(llm=llm, prompt=dynamic_prompt)

# Run the LLMChain with input_data
input_data = {"input": "tiger"}
response = chain.run(input_data)

print(response)

# Can also load Prompttemplate from json or yaml file
prompt_template.save("awesome_prompt.json")

from langchain.prompts import load_prompt
loaded_prompt = load_prompt("awesome_prompt.json")


# Another Example of FewShot Template

examples = [
    {
        "query": "How do I become a better programmer?",
        "answer": "Try talking to a rubber duck; it works wonders."
    }, {
        "query": "Why is the sky blue?",
        "answer": "It's nature's way of preventing eye strain."
    }
]
# The examples are usually in a dictionary with key-answer pairs

example_template = """
User: {query}
AI: {answer}
""" 

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative and funny responses to users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """


few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

#tools: List of tools the agent will have access to, used to format the prompt.
#prefix: String to put before the list of tools.
#suffix: String to put after the list of tools.
#input_variables: List of input variables the final prompt will expect

# Create the LLMChain for the few_shot_prompt_template
chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

# Run the LLMChain with input_data
input_data = {"query": "How can I learn quantum computing?"}
response = chain.run(input_data)

print(response)

'''
The FewShotPromptTemplate provided in the example demonstrates the power of dynamic prompts. Instead of using a static template, this approach incorporates examples of previous interactions, allowing the AI better to understand the context and style of the desired response.

Dynamic prompts offer several advantages over static templates:

Improved context understanding: By providing examples, the AI can grasp the context and style of responses more 
effectively, enabling it to generate responses that are more in line with the desired output.

Flexibility: Dynamic prompts can be easily customized and adapted to specific use cases, allowing developers to experiment with different prompt structures and find the most effective format for their application.

Better results: As a result of the improved context understanding and flexibility, dynamic prompts often yield higher-quality outputs that better match user expectations.


helps in optimizing token usage and managing the balance between the number of examples and prompt size

'''


# Assume there are lots of examples to choose from aside from the ones above

# Instead of using all the examples in the list of dictonaries directly we use LengthBasedExampleSelector

from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.chat_models import ChatOpenAI


example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100  
)


# By using this the code dynamically selects and includes examples based on their lengthand ensures the final prompt stays within desired token limit


dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=example_prompt,
    prefix=prefix, #Prefix and suffix come from above example
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)


llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Existing example and prompt definitions, and dynamic_prompt_template initialization

# Create the LLMChain for the dynamic_prompt_template
chain = LLMChain(llm=llm, prompt=dynamic_prompt_template)

# Run the LLMChain with input_data
input_data = {"query": "Who invented the telephone?"}
response = chain.run(input_data)

print(response)



###################################################

# MANAGING OUTPUTS WITH OUTPUT PARSERS

# While language models only give textual outputs there is No guarantee that the output will be of the same format making it difficult to work with the data

#Output parsers help create a data structure to define the expectations from the ouput precisely

# Creating a thesaurus 

#Pydantic Output Parser

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")

    # Throw error in case of receiving a numbered-list from API
    @validator('words')
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field

parser = PydanticOutputParser(pydantic_object=Suggestions)

from langchain.prompts import PromptTemplate

template = """
Offer a list of suggestions to substitue the specified target_word based the presented context.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"], #Input variables instantialised later using format.prompt()
    partial_variables={"format_instructions": parser.get_format_instructions()} #Partial variables instantialsied straight away
)

model_input = prompt.format_prompt(
			target_word="behaviour",
			context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

from langchain.llms import OpenAI

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
model = OpenAI(model_name='text-davinci-003', temperature=0.0)

output = model(model_input.to_string())

parser.parse(output)
#The parser object’s parse() function will convert the model’s string response to the format we specified. There is a list of words that you can index through and use in your applications.

# COMMA SEPERATED OUTPUT PARSER

# handles one specific case: anytime you want to receive a list of outputs from the model

from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

# Does not require setting up and is therefore less flexible

# Prepare the Prompt
template = """
Offer a list of suggestions to substitute the word '{target_word}' based the presented the following text: {context}.
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format(
  target_word="behaviour",
  context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)
# the use of .format() instead of .format_prompt() to generate the model’s input. The main difference compared to the previous subsection’s code is that we no longer need to call the .to_string() object since the prompt is already in string type

# Loading OpenAI API
model = OpenAI(model_name='text-davinci-003', temperature=0.0)

# Send the Request
output = model(model_input)
parser.parse(output)

## Structured Output Parser:

from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="words", description="A substitue word based on context"),
    ResponseSchema(name="reasons", description="the reasoning of why this word fits the context.")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

## FIXING ERRORS:

# The following approaches work with the PydanticOutputParser class since it is the only one with a validation method.

# Creating an output fixing parser

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

parser = PydanticOutputParser(pydantic_object=Suggestions)

missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}' 

parser.parse(missformatted_output) # The word reasoning is used instead of reason which willo throw an error

from langchain.llms import OpenAI
from langchain.output_parsers import OutputFixingParser

model = OpenAI(model_name='text-davinci-003', temperature=0.0)

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
outputfixing_parser.parse(missformatted_output)

# The from_llm() function takes the old parser and a language model as input parameters. Then, It initializes a new parser for you that has the ability to fix output errors. In this case, it successfully identified the misnamed key and changed it to what we defined.

## RETRYOUTPUTPARSER

# In some cases, the parser needs access to both the output and the prompt to process the full context
 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

parser = PydanticOutputParser(pydantic_object=Suggestions)

# Define prompt
template = """
Offer a list of suggestions to substitue the specified target_word based the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(target_word="behaviour", context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson.")

# Define Model
model = OpenAI(model_name='text-davinci-003', temperature=0.0)


from langchain.output_parsers import RetryWithErrorOutputParser

missformatted_output = '{"words": ["conduct", "manner"]}'

retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)

retry_parser.parse_with_prompt(missformatted_output, model_input) # Takes put output and model input to gain better context 

###############################################

## Creating knowledge graphs from textual data

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.graphs.networkx_graph import KG_TRIPLE_DELIMITER

# Prompt template for knowledge triple extraction
_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the text."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object. The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property.\n\n"
    "EXAMPLE\n"
    "It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
    f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "I'm going to the store.\n\n"
    "Output: NONE\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
    f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "{text}"
    "Output:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)

# Make sure to save your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
# Instantiate the OpenAI model
llm = OpenAI(model_name="text-davinci-003", temperature=0.9)

# Create an LLMChain using the knowledge triple extraction prompt
chain = LLMChain(llm=llm, prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT)

# Run the chain with the specified text
text = "The city of Paris is the capital and most populous city of France. The Eiffel Tower is a famous landmark in Paris."
triples = chain.run(text)

print(triples)


def parse_triples(response, delimiter=KG_TRIPLE_DELIMITER):
    if not response:
        return []
    return response.split(delimiter)

triples_list = parse_triples(triples)

# Print the extracted relation triplets
print(triples_list)