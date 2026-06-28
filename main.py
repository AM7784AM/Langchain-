import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


load_dotenv()

def main():
    print("Hello from langchain-course!")

    information = """Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX..."""

    summary_template = """
    Given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # Use Ollama locally
    llm = ChatOllama(model="gemma", temperature=0)
    llm = ChatOllama(model="llama3", temperature=0)

    chain = summary_prompt_template | llm
    
    print("Invoking LangChain with Ollama...")
    response = chain.invoke(input={"information": information})
    print("\n--- Model Response ---")
    print(response.content)

if __name__ == "__main__":
    main()
