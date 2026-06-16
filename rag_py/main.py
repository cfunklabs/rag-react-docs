from dotenv import load_dotenv
from langchain_core import __version__ as langchain_core_version
from langgraph.version import __version__ as langgraph_version
from langchain_anthropic import __version__ as langchain_anthropic_version
from langchain_anthropic import ChatAnthropic

load_dotenv()

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def print_versions():
    print(f"  - langchain_core_version: {langchain_core_version}")
    print(f"  - langgraph_version: {langgraph_version}")
    print(f"  - langchain_anthropic_version: {langchain_anthropic_version}")

def llm_health_check():
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)
    response = llm.invoke("Say this exact phrase, no more and no less: 'Claude API interface is up and running'")
    return True if response.content.lower() == "claude api interface is up and running" else False

def startup_check():
    print("Versions:")
    print_versions()
    print("\nChecking LLM service health...", end="")
    is_llm_healthy = llm_health_check()
    print(f" {GREEN if is_llm_healthy else RED}{'PASSED' if is_llm_healthy else 'FAILED'}{RESET}")
    return is_llm_healthy

def main():
    if not startup_check():
        print(f"{RED}Startup check failed. Exiting...{RESET}")
        return

if __name__ == "__main__":
    main()
