The chains are responsible for creating an end-to-end pipeline for using the language models.

They will join the model, prompt, memory, parsing output, and debugging capability and provide an easy-to-use interface.

A chain will 1) receive the user’s query as an input, 2) process the LLM’s response, and lastly, 3) return the output to the user.

```py
from langchain import PromptTemplate, OpenAI, LLMChain

prompt_template = "What is a word to replace the following: {word}?"

# Set the "OPENAI_API_KEY" environment variable before running following line.
llm = OpenAI(model_name="text-davinci-003", temperature=0)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

llm_chain("artificial")

```

The code above puts together the prompt and llm to create a chain that can do a required task

It is also possible to use the .apply() method to pass multiple inputs at once and receive a list for each input. The sole difference lies in the exclusion of inputs within the returned list. Nonetheless, the returned list will maintain the identical order as the input.

```py
input_list = [
    {"word": "artificial"},
    {"word": "intelligence"},
    {"word": "robot"}
]

llm_chain.apply(input_list)

```

The next method we will discuss is .predict(). (which could be used interchangeably with .run()) Its best use case is to pass multiple inputs for a single prompt. However, it is possible to use it with one input variable as well.

```py

prompt_template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=prompt_template, input_variables=["word", "context"]))

llm_chain.predict(word="fan", context="object")
# or llm_chain.run(word="fan", context="object")

```

# Parsers

We need a parser to fulfil the end-to-end pipeline in order to extract information from the LLM output.

Without a parser it would be difficult to maintain the output format we get back from the LLM to stay consistent

When using with a chain you simply specify the parser

```py
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
template = """List all possible words as substitute for 'artificial' as comma separated."""

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=template, output_parser=output_parser, input_variables=[])
    ) # don't know why two output parsers are used here

llm_chain.predict()

```

# Conversational Chain with Memory

Langchain provides a conversationalChain to track previous prompts and responses using the ConversationalBufferMemory class

```py

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

output_parser = CommaSeparatedListOutputParser()
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

conversation.predict(input="List all possible words as substitute for 'artificial' as comma separated.")

```

# Sequential Chain

A sequential Chain concatenates multiple chains into one

The SimpleSequentialChain: The simplest form of sequential chains, where each step has a singular input/output, and the output of one step is the input to the next.

SequentialChain: A more general form of sequential chains, allowing for multiple inputs/outputs.

#Custom Chain

A custom chain can be used to carry out any custom task

It starts by defining a classthat inherits from Chain Class

Three methods must be declared. The input_keys and output_keys which lets the model know what it should expect and a \_call method runs each chain and merges their outputs

EXAMPLE 1:

```py
from langchain.chains import LLMChain
from langchain.chains.base import Chain

from typing import Dict, List


class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}


prompt_1 = PromptTemplate(
    input_variables=["word"],
    template="What is the meaning of the following word '{word}'?",
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
    input_variables=["word"],
    template="What is a word to replace the following: {word}?",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
concat_output = concat_chain.run("artificial")
print(f"Concatenated output:\n{concat_output}")

```

EXAMPLE 2:

```py

class MyCustomChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}


```
