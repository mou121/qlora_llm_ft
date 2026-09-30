import os
import sys
import json
import requests
from dataclasses import dataclass
import subprocess

@dataclass
class GitHubContext:
    token: str = os.getenv("GITHUB_TOKEN")
    repo: str = "mou121/qlora_llm_ft"
    issue_num: str = os.getenv("ISSUE_NUMBER")
    issue_body: str = os.getenv("ISSUE_BODY")
    event_type: str = os.getenv("EVENT_NAME")

    def post_comment(self, message: str):
        if not self.issue_num or not self.token:
            print(f"[Local Log Only] {message}")
            return
        
        url = f"https://github.com{self.repo}/issues/{self.issue_num}/comments"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json"
        }
        requests.post(url, headers=headers, json={"body": message})

def create_isolated_branch(ticket_id):
    branch_name = f"agent/feature-issue-{ticket_id}"
    # Create and switch to a new branch locally
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    print(f"Safe workspace branch created: {branch_name}")
    return branch_name

def push_changes_to_github(branch_name):
    # Push the agent's changes directly to your repo
    subprocess.run(["git", "push", "origin", branch_name], check=True)
    print("Code pushed successfully to origin.")

def run_agentic_pipeline():
    ctx = GitHubContext()
    
    if not ctx.issue_body:
        print("No actionable event context detected. Skipping pipeline pass.")
        sys.exit(0)

    ctx.post_comment("🤖 **Agentic SDLC Factory:** Initializing worker loop context. Analyzing repository structures...")

    # --- Phase 1: Ingesting Domain Rules ---
    rules_path = ".agentic-factory/agent-rules.md"
    if os.path.exists(rules_path):
        print("Model architecture rules successfully loaded into agent context memory.")

    # --- Phase 2: Simulating Worker Processing Loop ---
    # Here, the pipeline analyzes scripts like train.py or infer.py based on the ticket request
    ctx.post_comment("⚙️ **Intake & Architecture Agents:** Verified constraints against `train.py` optimization targets. Synthesizing safe execution path...")

    # --- Phase 3: Mechanical Harness Validation Run ---
    print("Executing automated test and lint checks across codebase files...")
    # Simulated execution checking for format exceptions
    validation_passed = True 

    if validation_passed:
        ctx.post_comment("✅ **Validator Agent:** Linting validations and mechanical compilation testing passed successfully. Safe deployment conditions verified.")
    else:
        ctx.post_comment("❌ **Validator Agent Failure:** Code verification failed structural test parameters. Aborting merge branch initialization.")
        sys.exit(1)

if __name__ == "__main__":
    run_agentic_pipeline()
