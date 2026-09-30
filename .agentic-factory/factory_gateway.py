import os
import sys
import json
import requests
from dataclasses import dataclass
import subprocess
import traceback  # Crucial for capturing full error logs

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
        try:
            res = requests.post(url, headers=headers, json={"body": message})
            print(f"[GitHub API Status]: {res.status_code}")
        except Exception as e:
            print(f"Failed to transmit comment: {e}")

def generate_code_with_groq(context: GitHubContext, rules_content: str) -> str:
    """
    Calls Groq API safely using the official OpenAI sdk library bindings.
    """
    from openai import OpenAI
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment secret is missing or empty.")

    client = OpenAI(
        base_url="https://groq.com",
        api_key=api_key
    )

    existing_code = ""
    if os.path.exists("train.py"):
        with open("train.py", "r") as f:
            existing_code = f.read()

    system_prompt = f"""
    You are an automated code generator for an agentic SDLC pipeline.
    Write clean, production-ready Python code to solve the issue.
    
    RULES:
    {rules_content}
    
    OUTPUT FORMAT:
    - Return ONLY executable Python code.
    - Do NOT wrap code inside markdown blocks like ```python ... ```.
    - Do NOT give conversational text or explanations.
    """

    user_prompt = f"""
    Issue Title: {context.issue_title}
    Issue Description: {context.issue_body}

    Current train.py content:
    ---
    {existing_code if existing_code else "# train.py is currently empty"}
    ---

    Generate the complete new version of train.py integrating the request cleanly.
    """

    completion = client.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )

    raw_code = completion.choices.message.content
    
    # Clean up formatting safely
    if "```python" in raw_code:
        raw_code = raw_code.split("```python")[1].split("```")[0]
    elif "```" in raw_code:
        raw_code = raw_code.split("```")[1].split("```")[0]
        
    return raw_code.strip()

def create_isolated_branch(ticket_id):
    branch_name = f"agent/feature-issue-{ticket_id}"
    
    # Configure workspace identity parameters
    subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(["git", "config", "--global", "user.email", "github-actions[bot]@://github.com"], check=True)
    
    # Switch out of detached HEAD states natively
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    print(f"Safe workspace branch created locally: {branch_name}")
    return branch_name

def commit_and_push(branch_name, ticket_id):
    subprocess.run(["git", "add", "train.py"], check=True)
    
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not status.stdout.strip():
        print("No structural changes detected. Skipping commit generation step.")
        return

    subprocess.run(["git", "commit", "-m", f"feat(agent): dynamically generated patch via Groq for #{ticket_id}"], check=True)
    subprocess.run(["git", "push", "origin", branch_name, "--force"], check=True)

def open_pull_request(ctx, branch_name):
    url = f"https://github.com{ctx.repo}/pulls"
    headers = {
        "Authorization": f"Bearer {ctx.token}", 
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "title": f"Agent Resolve: {ctx.issue_title} (#{ctx.issue_num})",
        "head": branch_name,
        "base": "main",
        "body": f"Automated feature deployment powered by Groq Open Source Inference Core Engine.\n\nCloses #{ctx.issue_num}."
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 201:
        ctx.post_comment(f"🚀 **PR Created Successfully:** {res.json()['html_url']}")
    else:
        raise RuntimeError(f"GitHub Pull Request API returned error status {res.status_code}: {res.text}")

def run_agentic_pipeline():
    ctx = GitHubContext()
    print("--- PIPELINE START ENGINE LOG ---")
    
    if not ctx.issue_body:
        print("No issue payload context found. Exiting.")
        sys.exit(0)

    # 1. Initial Check-in Comment
    ctx.post_comment("🤖 **Agentic SDLC Factory:** Pipeline boot validation initialized. Processing repository code parameters...")

    try:
        # Load rules boundaries
        rules_content = ""
        if os.path.exists(".agentic-factory/agent-rules.md"):
            with open(".agentic-factory/agent-rules.md", "r") as f:
                rules_content = f.read()

        # 2. Workspace Branch Step
        print("Executing local Git branch creation parameters...")
        active_branch = create_isolated_branch(ctx.issue_num or "test-run")
        
        # 3. Code Generation Step
        print("Querying external open-source LLM engine inference layer...")
        generated_code = generate_code_with_groq(ctx, rules_content)
        
        # 4. File Modification Step
        with open("train.py", "w") as f:
            f.write(generated_code)
        print("Code successfully outputted onto local system disk.")

        # 5. Push and PR synchronization Steps
        print("Syncing git repositories upstream...")
        commit_and_push(active_branch, ctx.issue_num)
        open_pull_request(ctx, active_branch)
        
    except Exception as e:
        # --- CRITICAL ERROR CAPTURE SYSTEM ---
        # Extracts the entire system crash trajectory
        error_stack = traceback.format_exc()
        
        # Formulate a structured markdown bug trace report comment
        crash_report = f"""
❌ **Pipeline Processing Hard-Crash Exception Detected!**

The agentic execution loop broke before code serialization could map upstream.

**Error Summary:** `{str(e)}`
**Execution Diagnostic Traceback:**
```text
{error_stack}
```
"""
        # Send the exact crash trace directly into your open GitHub issue thread
        ctx.post_comment(crash_report)
        print(f"Pipeline crashed. Transmitted Diagnostic crash report to issue thread:\n{error_stack}")
        sys.exit(1)

if __name__ == "__main__":
    run_agentic_pipeline()
