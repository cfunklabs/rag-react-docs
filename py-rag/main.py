from dotenv import load_dotenv
from langchain_core import __version__ as langchain_core_version
from langgraph.version import __version__ as langgraph_version
from langchain_anthropic import __version__ as langchain_anthropic_version
from langchain_anthropic import ChatAnthropic

load_dotenv()

print(f"langchain_core_version: {langchain_core_version}")
print(f"langgraph_version: {langgraph_version}")
print(f"langchain_anthropic_version: {langchain_anthropic_version}")

def main():
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)
    response = llm.invoke("Hello, tell me that you are up and running.")
    print(response.content)

if __name__ == "__main__":
    main()
