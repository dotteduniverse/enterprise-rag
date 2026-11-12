from pydantic import BaseModel, Field, ValidationError
from typing import Literal

# 1. Define the strict schema
class RefundRequest(BaseModel):
    order_id: str = Field(..., description="Must start with ORD-")
    amount: float = Field(..., gt=0, le=500.0, description="Max refund is $500")
    reason: Literal["damaged", "wrong_item", "late_delivery"]

# 2. The Tool Execution Function
def execute_refund_tool(llm_arguments: dict) -> dict:
    """
    This is what the Agent calls when the LLM decides to use the refund tool.
    """
    try:
        # Validate the LLM's arguments against the strict schema
        valid_data = RefundRequest(**llm_arguments)
        
        # If we get here, the data is perfect. Simulate the actual refund.
        return {
            "status": "success", 
            "message": f"Successfully refunded ${valid_data.amount} for {valid_data.order_id}"
        }
        
    except ValidationError as e:
        # THE ERROR-AS-OBSERVATION PATTERN
        # We don't crash. We return the exact validation error to the LLM.
        error_detail = e.errors()[0]['msg']
        field_name = e.errors()[0]['loc'][0]
        
        return {
            "status": "error",
            "observation": f"Validation failed on '{field_name}': {error_detail}. Please fix your arguments and try again."
        }

# --- Test it out ---
if __name__ == "__main__":
    print("--- Test 1: The LLM tries to steal $1000 ---")
    bad_args = {"order_id": "ORD-123", "amount": 1000.0, "reason": "damaged"}
    print(execute_refund_tool(bad_args))
    
    print("\n--- Test 2: The LLM hallucinates a bad reason ---")
    bad_args_2 = {"order_id": "ORD-123", "amount": 50.0, "reason": "I just want my money back"}
    print(execute_refund_tool(bad_args_2))
    
    print("\n--- Test 3: The LLM gets it right ---")
    good_args = {"order_id": "ORD-123", "amount": 50.0, "reason": "damaged"}
    print(execute_refund_tool(good_args))