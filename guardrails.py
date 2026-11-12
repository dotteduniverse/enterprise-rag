'''
    action is the dictionary the LLM generated. It looks like this: 
    {"tool": "issue_refund", "arguments": {"amount": 2000, "user_id": "user_B"}}

session is the dictionary of the actual logged-in user. It looks like this: 
{"logged_in_user_id": "user_A"}

    '''

def policy_gate(action: dict, session: dict) -> str:
    # Extract arguments safely (default to empty dict if missing)
    args = action.get("arguments", {})
    
    # 1. Get the amount (default to 0 if missing so it doesn't crash)
    requested_amount = args.get("amount", 0)
    permitted_amount = 500
    
    # 2. Get the target user the LLM wants to refund
    requested_user_id = args.get("user_id")
    
    # 3. Get the actual logged-in user
    session_user_id = session.get("logged_in_user_id")
    
    # 4. The Check
    if (requested_amount > permitted_amount) or (session_user_id != requested_user_id):
        return "BLOCK"
    
    return "ALLOW"