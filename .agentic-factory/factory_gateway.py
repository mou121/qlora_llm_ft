import os
import sys
import json
import requests
from dataclasses import dataclass
import subprocess
import traceback

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
        
        # FIXED: Re-mapped to canonical modern bearer authorization format to clear 404 blocks
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

def generate_code_with_groq_sdk(context: GitHubContext, rules_content: str) -> str:
    """
    Uses the official Groq SDK library wrapper to guarantee clean HTTP transmission.
    """
    # Force install the official library dynamically if it isn't in requirements yet
    try:
        from groq import Groq
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "groq"], check=True)
        from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL: GROQ_API_KEY environment secret is missing or empty.")

    # Initialize client through official Groq bindings
    client = Groq(api_key=api_key)

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

    # Native SDK completion loop invocation using official model parameters
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )

    raw_code = completion.choices[0].message.content
    
    # Strip any markdown structural strings cleanly
    clean_code = raw_code.replace("```python", "").replace("```", "")
    return clean_code.strip()

def create_isolated_branch(ticket_id):
    branch_name = f"agent/feature-issue-{ticket_id}"
    subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(["git", "config", "--global", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
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
    """
    Create a pull request for the pushed agent branch.

    Handles:
    - Existing PR for the same head and base
    - Successful PR creation
    - No new differences (422)
    - Other GitHub API errors
    """

    repo = ctx.repo.strip("/")
    owner, repo_name = repo.split("/", 1)

    base_branch = "main"
    head_branch = f"{owner}:{branch_name}"

    api_url = f"https://api.github.com/repos/{repo}/pulls"

    headers = {
        "Authorization": f"Bearer {ctx.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Shared PR metadata
    title = f"Agent Resolve: {ctx.issue_title} (#{ctx.issue_num})"
    body = (
        "Automated feature deployment powered by the Groq SDK.\n\n"
        f"Closes #{ctx.issue_num}."
    )

    # 1. Check whether an open PR already exists for this branch.
    existing_pr_url = f"{api_url}?state=open&head={head_branch}&base={base_branch}"

    try:
        existing_res = requests.get(
            existing_pr_url,
            headers=headers,
            timeout=30,
        )
        existing_res.raise_for_status()
        existing_prs = existing_res.json()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to check existing pull requests: {exc}"
        ) from exc

    if existing_prs:
        pr = existing_prs[0]
        pr_html_url = pr.get("html_url")

        print(f"An active PR already exists: {pr_html_url}")

        ctx.post_comment(
            f"ℹ️ **Pull Request Already Exists:** {pr_html_url}\n\n"
            f"Branch: `{branch_name}` → `{base_branch}`"
        )
        return pr

    # 2. Create the PR if no active PR was found.
    payload = {
        "title": title,
        "head": head_branch,
        "base": base_branch,
        "body": body,
    }

    try:
        res = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30,
        )

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to send pull request creation request: {exc}"
        ) from exc

    # 3. Handle successful creation.
    if res.status_code == 201:
        pr = res.json()
        pr_html_url = pr.get("html_url")

        print(f"PR Created Successfully: {pr_html_url}")

        ctx.post_comment(
            f"🚀 **PR Created Successfully:** {pr_html_url}"
        )
        return pr

    # 4. Handle GitHub validation errors.
    if res.status_code == 422:
        try:
            error_data = res.json()
        except ValueError:
            error_data = {"message": res.text}

        error_message = error_data.get("message", "Validation failed")
        errors = error_data.get("errors", [])

        # GitHub can return 422 if a PR already exists or
        # if the branches have no differences.
        print(f"PR creation returned 422: {error_message}")
        print(f"GitHub error details: {errors}")

        # Re-check in case another process created the PR
        # between our initial check and this POST request.
        try:
            retry_res = requests.get(
                existing_pr_url,
                headers=headers,
                timeout=30,
            )
            retry_res.raise_for_status()
            retry_prs = retry_res.json()

        except requests.RequestException as exc:
            raise RuntimeError(
                "PR creation returned 422, and the follow-up "
                f"PR lookup failed: {exc}. Original response: {res.text}"
            ) from exc

        if retry_prs:
            pr = retry_prs[0]
            pr_html_url = pr.get("html_url")

            print(f"Found existing PR after 422: {pr_html_url}")

            ctx.post_comment(
                f"ℹ️ **Pull Request Already Exists:** {pr_html_url}"
            )
            return pr

        # No open PR found. Don't claim that a PR exists:
        # the remaining likely explanation is no diff or
        # another validation issue.
        ctx.post_comment(
            "⚠️ **PR Could Not Be Created**\n\n"
            f"GitHub returned HTTP 422: `{error_message}`\n\n"
            f"Branch `{branch_name}` was pushed, but no matching "
            f"open PR was found for `{base_branch}`. "
            "Check whether the branch has changes compared with "
            "the base branch and review GitHub's validation details."
        )

        return None

    # 5. Raise for authentication, permission, missing repo,
    # rate-limit, and other unexpected API failures.
    raise RuntimeError(
        "GitHub Pull Request API returned "
        f"HTTP {res.status_code}: {res.text}"
    )

def run_agentic_pipeline():
    ctx = GitHubContext()
    print("--- PIPELINE NATIVE START ENGINE LOG ---")
    
    if not ctx.issue_body:
        print("No issue payload context found. Exiting.")
        sys.exit(0)

    try:
        rules_content = ""
        if os.path.exists(".agentic-factory/agent-rules.md"):
            with open(".agentic-factory/agent-rules.md", "r") as f:
                rules_content = f.read()

        print("Executing local Git branch creation parameters...")
        active_branch = create_isolated_branch(ctx.issue_num or "test-run")
        
        print("Querying Groq SDK endpoint framework...")
        generated_code = generate_code_with_groq_sdk(ctx, rules_content)
        
        with open("train.py", "w") as f:
            f.write(generated_code)
        print("Code successfully outputted onto local system disk.")

        print("Syncing git repositories upstream...")
        commit_and_push(active_branch, ctx.issue_num)
        
        open_pull_request(ctx, active_branch)
        
    except Exception as e:
        error_stack = traceback.format_exc()
        print(f"Pipeline crashed. Diagnostic stack:\n{error_stack}")
        ctx.post_comment(f"❌ **Pipeline Failure Event:** `{str(e)}`\n\n```text\n{error_stack}\n```")
        sys.exit(1)

if __name__ == "__main__":
    run_agentic_pipeline()
