from llm_router import get_llm_client
from schemas import Provider
from tools import execute_refund_tool
import json

from guardrails import policy_gate
def run_agent(user_query: str, history: list = None, max_steps: int = 3):
    client, model = get_llm_client(Provider.GROQ)
    
    system_prompt = """
    You are a customer support agent. 
    You have access to one tool: 'issue_refund'.
    It requires: order_id (string), amount (float), reason (string: 'damaged', 'wrong_item', or 'late_delivery').
    
    If the user asks for a refund, output EXACTLY a JSON object like this:
    {"tool": "issue_refund", "arguments": {"order_id": "ORD-123", "amount": 50.0, "reason": "damaged"}}
    
    If no tool is needed, just output normal text.
    """
    
    print(f"👤 User: {user_query}\n")
    
    
    # If history is None, make it an empty list
    if history is None:
        history = []
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)  # Add the past conversation
    messages.append({"role": "user", "content": user_query}) # Add the new question
    
    steps = 0
    
    while steps < max_steps:
        steps += 1
        print(f"--- Step {steps} ---")
        
        # 1. THINK
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        
        llm_output = response.choices[0].message.content
        print(f"🤖 LLM: {llm_output[:100]}...") # Truncate for clean output
        
        # 2. ACT (Check for tool call)
        try:
            tool_call = json.loads(llm_output)
            if tool_call.get("tool") == "issue_refund":
                messages.append({"role": "assistant", "content": llm_output})
                session = {"logged_in_user_id": "user_A"}
                print(f"🛑 [POLICYGATE] Executing guardrail for the user...")
                if policy_gate(tool_call,session) == "BLOCK":
                    print(f"🛑 [POLICYGATE] session user mismatch, invalid session..")
                    break
                print(f" [POLICYGATE] Session user validated...")             
                
                # 3. EXECUTE TOOL
                print(f"⚙️ [ORCHESTRATOR] Executing tool...")
                tool_result = execute_refund_tool(tool_call["arguments"])
                print(f"🛠️ [TOOL RESULT]: {tool_result}")
                
                # 4. OBSERVE
                if tool_result["status"] == "success":
                    print("\n✅ Final Answer ready.")
                    messages.append({"role": "user", "content": f"Tool Result: {json.dumps(tool_result)}. Tell the user it succeeded."})
                    break # We are done
                else:
                    print("🔄 [ORCHESTRATOR] Tool failed. Feeding error back to LLM for self-correction.")
                    messages.append({"role": "user", "content": f"Tool Error: {tool_result['observation']}. Please fix the arguments and try again."})
                    # The loop will continue, and the LLM will try again!
        except json.JSONDecodeError:
            # The LLM gave a final text answer
            print(f"\n✅ Final Answer: {llm_output}")
            break
            
    if steps >= max_steps:
        print("\n🛑 [ORCHESTRATOR] Max steps reached. Handing off to human.")

if __name__ == "__main__":
    # TEST: Let's force the LLM to fail and self-correct.
    # We ask for a refund with a bad reason ("I don't like it").
    #run_agent("I need a refund for ORD-555. The amount is 200 dollars. The reason is 'I don't like it'.")
    #run_agent("I need a refund for ORD-555. The amount is 2000 dollars. The reason is 'damaged'.")
    my_history = [
    {"role": "user", "content": "I want a refund for order ORD-555."},
    {"role": "assistant", "content": "What is the reason for the refund?"}
    ]
    run_agent("The reason is 'damaged' and the amount is 50 dollars.", history=my_history)