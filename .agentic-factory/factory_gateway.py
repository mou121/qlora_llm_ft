import os
import sys
import json
import requests
from dataclasses import dataclass
import subprocess
from openai import OpenAI  # Used to call open-source API providers

@dataclass
class GitHubContext:
    token: str = os.getenv("GITHUB_TOKEN")
    repo: str = "mou121/qlora_llm_ft"
    issue_num: str = os.getenv("ISSUE_NUMBER")
    issue_title: str = os.getenv("ISSUE_TITLE", "Feature Request")
    issue_body: str = os.getenv("ISSUE_BODY")
    event_type: str = os.getenv("EVENT_NAME")

    def post_comment(self, message: str):
        if not self.issue_num or not self.token:
            print(f"[Local Log Only] {message}")
            return
        url = f"https://github.com/repos{self.repo}/issues/{self.issue_num}/comments"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        requests.post(url, headers=headers, json={"body": message})

def generate_code_with_open_llm(context: GitHubContext, rules_content: str) -> str:
    """
    Calls an open-source LLM via a provider API endpoint (Groq/TogetherAI/Ollama).
    """
    # 1. Initialize client using Open-Source endpoints
    # For Groq: base_url="https://groq.com", api_key=os.getenv("GROQ_API_KEY")
    # For Ollama (Local): base_url="http://localhost:11434/v1", api_key="ollama"
    
    api_key = os.getenv("OPEN_LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    base_url = os.getenv("OPEN_LLM_BASE_URL") or "https://groq.com"
    model_name = os.getenv("OPEN_LLM_MODEL") or "llama-3.1-70b-versatile"

    if not api_key:
        print("Missing OPEN_LLM_API_KEY or GROQ_API_KEY. Falling back to default baseline generator.")
        return "# Baseline fallback\nprint('No LLM API Key provided')"

    client = OpenAI(base_url=base_url, api_key=api_key)

    # 2. Extract existing target code structure if it exists
    existing_code = ""
    if os.path.exists("train.py"):
        with open("train.py", "r") as f:
            existing_code = f.read()

    # 3. Formulate system prompting constraints
    system_prompt = f"""
    You are an automated software architecture engine for an Agentic SDLC Pipeline.
    Your task is to write clean Python code to resolve the user's issue description.
    
    CRITICAL PROJECT RULES:
    {rules_content}
    
    OUTPUT FORMAT INSTRUCTIONS:
    - Respond with RAW, executable Python code only.
    - Do NOT include markdown code blocks (```python ... ```).
    - Do NOT include conversational text, pleasantries, or explanations.
    """

    user_prompt = f"""
    Issue Title: {context.issue_title}
    Issue Description: {context.issue_body}

    Current train.py codebase content:
    ---
    {existing_code if existing_code else "# train.py is currently empty"}
    ---

    Generate the complete, updated version of train.py integrating the request.
    """

    # 4. Execute the Open-Source LLM Generation Inference Pass
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1  # Low temperature for strict, stable deterministic output
    )

    return completion.choices[0].message.content

def create_isolated_branch(ticket_id):
    branch_name = f"agent/feature-issue-{ticket_id}"
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(["git", "config", "user.email", "github-actions[bot]@://github.com"], check=True)
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    return branch_name

def commit_and_push(branch_name, ticket_id):
    subprocess.run(["git", "add", "train.py"], check=True)
    subprocess.run(["git", "commit", "-m", f"feat(agent): dynamically generated patch for #{ticket_id}"], check=True)
    subprocess.run(["git", "push", "origin", branch_name, "--force"], check=True)

def open_pull_request(ctx, branch_name):
    url = f"https://github.com{ctx.repo}/pulls"
    headers = {"Authorization": f"Bearer {ctx.token}", "Accept": "application/vnd.github+json"}
    payload = {
        "title": f"Agent Resolve: {ctx.issue_title} (#{ctx.issue_num})",
        "head": branch_name,
        "base": "main",
        "body": f"Automated feature generation powered by Open Source LLM ({os.getenv('OPEN_LLM_MODEL', 'Llama-3')}).\n\nCloses #{ctx.issue_num}."
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 201:
        ctx.post_comment(f"🚀 **PR Created:** {res.json()['html_url']}")

def run_agentic_pipeline():
    ctx = GitHubContext()
    if not ctx.issue_body:
        print("No issue payload context found.")
        sys.exit(0)

    # Load agent-rules boundaries
    rules_content = ""
    if os.path.exists(".agentic-factory/agent-rules.md"):
        with open(".agentic-factory/agent-rules.md", "r") as f:
            rules_content = f.read()

    ctx.post_comment("🤖 **Agentic SDLC Factory:** Querying Open Source LLM to synthesize code modifications...")

    try:
        active_branch = create_isolated_branch(ctx.issue_num or "test-run")
        
        # Call the open-source LLM engine
        generated_code = generate_code_with_open_llm(ctx, rules_content)
        
        # Overwrite file with the model's output patch safely
        with open("train.py", "w") as f:
            f.write(generated_code)

        commit_and_push(active_branch, ctx.issue_num)
        open_pull_request(ctx, active_branch)
    except Exception as e:
        ctx.post_comment(f"❌ **Pipeline Loop Exception:** {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_agentic_pipeline()
