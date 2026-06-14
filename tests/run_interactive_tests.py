#!/usr/bin/env python3
"""
CloudDash Support — Live Interactive API Test Runner
Loads tests/test_payloads.json and executes requests turn-by-turn
against the live running server (default: http://127.0.0.1:8001).
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# ANSI colors for beautiful terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

DEFAULT_BASE_URL = "http://127.0.0.1:8001"

def print_banner():
    print(f"\n{Colors.BOLD}{Colors.HEADER}================================================================{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}       ⚡ CLOUDDASH MULTI-AGENT SYSTEM INTERACTIVE TESTER ⚡{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}================================================================{Colors.ENDC}")

def check_server(base_url):
    try:
        r = requests.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            status = r.json()
            print(f"{Colors.GREEN}✓ Live server connected at {base_url}!{Colors.ENDC}")
            print(f"  - Health Status: {status.get('status')}")
            print(f"  - KB Loaded: {status.get('kb_loaded')}")
            print(f"  - Agents Ready: {status.get('agents_ready')}\n")
            return True
    except requests.exceptions.ConnectionError:
        pass
    print(f"{Colors.WARNING}⚠ Could not connect to live server at {base_url}.{Colors.ENDC}")
    print(f"  Please make sure your API is running. Running: uvicorn api.main:app --host 127.0.0.1 --port 8001")
    return False

def load_payloads():
    file_path = Path(__file__).resolve().parent / "test_payloads.json"
    if not file_path.exists():
        print(f"{Colors.FAIL}Error: test_payloads.json not found in {file_path.parent}!{Colors.ENDC}")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_scenario(scenario, base_url):
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ Running Scenario: {scenario['name']}{Colors.ENDC}")
    print(f"{Colors.CYAN}Description: {scenario['description']}{Colors.ENDC}")
    print("-" * 64)

    conv_id = None
    
    for turn_info in scenario["flow"]:
        turn = turn_info["turn"]
        endpoint = turn_info["endpoint"]
        payload = turn_info["payload"]
        
        print(f"\n{Colors.BOLD}{Colors.UNDERLINE}Turn {turn}: {endpoint}{Colors.ENDC}")
        
        # Build URL and JSON Payload
        if "POST /conversations" in endpoint and "/messages" not in endpoint:
            url = f"{base_url}/conversations"
            req_payload = payload
        else:
            if not conv_id:
                print(f"{Colors.FAIL}Error: No active conversation ID from previous turns!{Colors.ENDC}")
                return
            url = f"{base_url}/conversations/{conv_id}/messages"
            req_payload = {
                "conversation_id": conv_id,
                "message": payload["message"]
            }

        print(f"{Colors.BOLD}Request Payload:{Colors.ENDC}")
        print(json.dumps(req_payload, indent=2))
        
        input(f"\nPress Enter to send request... ")
        print(f"{Colors.CYAN}Sending HTTP POST request...{Colors.ENDC}")
        
        start_time = time.time()
        try:
            r = requests.post(url, json=req_payload, timeout=120)
            elapsed = time.time() - start_time
            
            print(f"{Colors.GREEN}Response received in {elapsed:.2f} seconds. Status Code: {r.status_code}{Colors.ENDC}")
            
            if r.status_code != 200:
                print(f"{Colors.FAIL}Error response:{Colors.ENDC}")
                print(r.text)
                return
                
            res_data = r.json()
            conv_id = res_data.get("conversation_id")
            trace_id = res_data.get("trace_id")
            
            print(f"\n{Colors.BOLD}Response Fields:{Colors.ENDC}")
            print(f"  • {Colors.BOLD}Conversation ID:{Colors.ENDC} {Colors.YELLOW if conv_id else Colors.FAIL}{conv_id}{Colors.ENDC}")
            print(f"  • {Colors.BOLD}Trace ID:{Colors.ENDC} {trace_id}")
            print(f"  • {Colors.BOLD}Current Agent:{Colors.ENDC} {Colors.GREEN}{res_data.get('current_agent')}{Colors.ENDC}")
            print(f"  • {Colors.BOLD}Is Resolved:{Colors.ENDC} {res_data.get('is_resolved')}")
            print(f"  • {Colors.BOLD}Requires Human (Escalated):{Colors.ENDC} {Colors.FAIL if res_data.get('requires_human') else Colors.GREEN}{res_data.get('requires_human')}{Colors.ENDC}")
            
            if res_data.get("escalation_package"):
                print(f"  • {Colors.BOLD}Escalation Package:{Colors.ENDC}")
                print(json.dumps(res_data.get("escalation_package"), indent=4))
                
            citations = res_data.get("citations", [])
            if citations:
                print(f"  • {Colors.BOLD}Citations Used ({len(citations)}):{Colors.ENDC}")
                for cit in citations:
                    print(f"    - [{cit.get('article_id')}] {cit.get('title')} (Category: {cit.get('category')}, Relevancy: {cit.get('score'):.4f})")
            else:
                print(f"  • {Colors.BOLD}Citations Used:{Colors.ENDC} None")
                
            print(f"\n{Colors.BOLD}Agent Response Text:{Colors.ENDC}")
            print(f"{Colors.BLUE}{res_data.get('response')}{Colors.ENDC}")
            
            # Query Handovers if it was a message
            if turn > 1:
                try:
                    h = requests.get(f"{base_url}/conversations/{conv_id}/handovers", timeout=5)
                    h_data = h.json()
                    events = h_data.get("events", [])
                    if events:
                        print(f"\n{Colors.WARNING}⚡ HANDOVER DETECTED in Audit Log ({len(events)} events):{Colors.ENDC}")
                        for ev in events:
                            print(f"  {Colors.WARNING}→ {ev.get('source_agent')} to {ev.get('target_agent')} (Reason: {ev.get('reason')}){Colors.ENDC}")
                except Exception:
                    pass

            print("-" * 64)

        except requests.exceptions.Timeout:
            print(f"{Colors.FAIL}Error: Request timed out.{Colors.ENDC}")
            return
        except Exception as e:
            print(f"{Colors.FAIL}Error executing request: {e}{Colors.ENDC}")
            return
            
    print(f"\n{Colors.GREEN}✓ Completed all turns for: {scenario['name']}{Colors.ENDC}\n")

def main():
    print_banner()
    
    # Allow overriding base URL via environment or CLI arg
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    
    connected = check_server(base_url)
    
    scenarios = load_payloads()
    
    while True:
        print(f"\n{Colors.BOLD}Available Test Scenarios:{Colors.ENDC}")
        for idx, sc in enumerate(scenarios, 1):
            agent_tag = sc["flow"][0]["expected"].get("current_agent", "triage")
            print(f"  [{idx:2d}] {Colors.BOLD}{sc['name']}{Colors.ENDC}")
            print(f"       Role: {agent_tag.upper()} | {sc['description'][:90]}...")
            
        print(f"\n  [q] Quit the tester")
        
        choice = input(f"\nSelect a test case to run (1-{len(scenarios)} or q): ").strip().lower()
        if choice == 'q':
            print(f"\n{Colors.GREEN}Thanks for testing CloudDash! Goodbye.{Colors.ENDC}")
            break
            
        try:
            val = int(choice)
            if 1 <= val <= len(scenarios):
                if not connected:
                    confirm = input(f"{Colors.WARNING}We are not connected to the server. Try running anyway? (y/n): {Colors.ENDC}").strip().lower()
                    if confirm != 'y':
                        continue
                run_scenario(scenarios[val-1], base_url)
            else:
                print(f"{Colors.FAIL}Invalid option. Enter a number between 1 and {len(scenarios)}.{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.FAIL}Invalid input. Please enter a valid number or 'q'.{Colors.ENDC}")

if __name__ == "__main__":
    main()
