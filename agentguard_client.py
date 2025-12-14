"""
============================================================================
AGENTGUARD CLIENT SDK
============================================================================

Python client for integrating AgentGuard into agent frameworks.
Shows how OpenAI, LangChain, AutoGPT, etc. would use AgentGuard.

Installation:
    pip install requests

Usage:
    from agentguard_client import AgentGuardClient
    
    client = AgentGuardClient("http://localhost:5000")
    result = client.enforce(agent_id="my-agent", code="dangerous_code()")
    
    if result['decision'] == 'ALLOW':
        execute(code)
    else:
        handle_block(result)
"""

import requests
from typing import Dict, List, Any, Optional
import json


class AgentGuardClient:
    """
    Python client for AgentGuard API.
    
    Example:
        client = AgentGuardClient("http://localhost:5000")
        result = client.enforce(
            agent_id="agent-123",
            code="account.balance -= 100",
            context={"has_auth_check": False}
        )
    """
    
    def __init__(self, base_url: str = "http://localhost:5000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'
    
    def health_check(self) -> Dict[str, Any]:
        """Check if AgentGuard is healthy"""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()
    
    def get_contract(self) -> str:
        """Get product contract"""
        response = self.session.get(f"{self.base_url}/api/v1/contract")
        response.raise_for_status()
        return response.json()['contract']
    
    def analyze(self, agent_id: str, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze code without enforcing.
        Returns what WOULD happen.
        
        Returns:
            {
                'would_block': bool,
                'blocked_by': str or None,
                'violations': list,
                'layer_results': list
            }
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/analyze",
            json={
                'agent_id': agent_id,
                'code': code,
                'context': context or {}
            }
        )
        response.raise_for_status()
        return response.json()
    
    def enforce(self, 
                agent_id: str, 
                code: str, 
                session_id: str = None,
                language: str = "python",
                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enforce safety checks on code.
        
        Returns:
            {
                'decision': 'ALLOW' | 'BLOCK' | 'REQUIRES_APPROVAL',
                'blocked_by': str or None,
                'violations': list,
                'latency_ms': float,
                'layer_results': list,
                'audit_id': str
            }
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/enforce",
            json={
                'agent_id': agent_id,
                'session_id': session_id,
                'language': language,
                'code': code,
                'context': context or {}
            }
        )
        response.raise_for_status()
        return response.json()
    
    def execute(self, agent_id: str, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute code with enforcement (premium feature).
        Only executes if all layers pass.
        
        Returns:
            {
                'executed': bool,
                'decision': str,
                'output': str or None
            }
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/execute",
            json={
                'agent_id': agent_id,
                'code': code,
                'context': context or {}
            }
        )
        response.raise_for_status()
        return response.json()
    
    def approve(self, approval_id: str, approver: str, conditions: str = None) -> Dict[str, Any]:
        """Approve a pending request"""
        response = self.session.post(
            f"{self.base_url}/api/v1/approval/{approval_id}/approve",
            json={
                'approver': approver,
                'conditions': conditions
            }
        )
        response.raise_for_status()
        return response.json()
    
    def reject(self, approval_id: str, approver: str, reason: str) -> Dict[str, Any]:
        """Reject a pending request"""
        response = self.session.post(
            f"{self.base_url}/api/v1/approval/{approval_id}/reject",
            json={
                'approver': approver,
                'reason': reason
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        response = self.session.get(f"{self.base_url}/api/v1/metrics")
        response.raise_for_status()
        return response.text


# ============================================================================
# INTEGRATION EXAMPLES
# ============================================================================

def example_basic_usage():
    """Example: Basic enforcement"""
    client = AgentGuardClient()
    
    # Check if service is healthy
    health = client.health_check()
    print(f"AgentGuard Status: {health['status']}")
    
    # Enforce code
    result = client.enforce(
        agent_id="my-agent-001",
        code="User.objects.all().delete()",
        context={}
    )
    
    if result['decision'] == 'ALLOW':
        print("✅ Code allowed - safe to execute")
    elif result['decision'] == 'BLOCK':
        print(f"❌ Code blocked by {result['blocked_by']}")
        print(f"Violations: {result['violations']}")
    elif result['decision'] == 'REQUIRES_APPROVAL':
        print(f"🚨 Requires approval: {result['approval_id']}")


def example_langchain_integration():
    """Example: LangChain integration"""
    from langchain.agents import Tool, AgentExecutor
    from langchain.llms import OpenAI
    
    # Initialize AgentGuard client
    guard = AgentGuardClient()
    
    # Create safe execution wrapper
    def safe_execute(code: str) -> str:
        """Execute code only if AgentGuard allows"""
        result = guard.enforce(
            agent_id="langchain-agent",
            code=code
        )
        
        if result['decision'] == 'ALLOW':
            # Actually execute the code
            exec_result = guard.execute(
                agent_id="langchain-agent",
                code=code
            )
            return exec_result.get('output', 'Executed successfully')
        else:
            return f"BLOCKED: {result['violations']}"
    
    # Create LangChain tool with safety wrapper
    safe_tool = Tool(
        name="SafePythonREPL",
        func=safe_execute,
        description="Execute Python code with safety guarantees"
    )
    
    # Use in agent
    # agent = create_agent(tools=[safe_tool], ...)


def example_autogpt_integration():
    """Example: AutoGPT integration"""
    
    class SafeAutoGPTExecutor:
        """AutoGPT executor with AgentGuard"""
        
        def __init__(self, agent_id: str):
            self.guard = AgentGuardClient()
            self.agent_id = agent_id
        
        def execute_code(self, code: str, context: Dict = None):
            """Execute code with safety checks"""
            
            # Analyze first (optional - for logging)
            analysis = self.guard.analyze(
                agent_id=self.agent_id,
                code=code,
                context=context
            )
            
            if analysis['would_block']:
                print(f"⚠️ Code would be blocked: {analysis['violations']}")
            
            # Enforce
            result = self.guard.enforce(
                agent_id=self.agent_id,
                code=code,
                context=context
            )
            
            if result['decision'] == 'ALLOW':
                # Execute via guard (uses sandbox)
                return self.guard.execute(
                    agent_id=self.agent_id,
                    code=code,
                    context=context
                )
            else:
                raise SecurityError(f"Code blocked: {result['violations']}")
    
    # Usage
    executor = SafeAutoGPTExecutor(agent_id="autogpt-001")
    # executor.execute_code("import os; os.listdir()")


def example_openai_assistants_integration():
    """Example: OpenAI Assistants API integration"""
    
    def safe_code_interpreter(code: str, agent_id: str) -> Dict:
        """
        Wrapper for OpenAI code interpreter that adds AgentGuard safety.
        
        Instead of directly executing code, check with AgentGuard first.
        """
        guard = AgentGuardClient()
        
        # Check with AgentGuard
        result = guard.enforce(
            agent_id=agent_id,
            code=code,
            context={
                'environment': 'openai_assistant',
                'has_auth_check': True  # OpenAI has its own auth
            }
        )
        
        if result['decision'] != 'ALLOW':
            return {
                'success': False,
                'error': f"Safety check failed: {result['violations']}",
                'audit_id': result['audit_id']
            }
        
        # If allowed, forward to OpenAI
        # openai_result = openai.code_interpreter.execute(code)
        # return openai_result
        
        return {'success': True, 'output': 'Would execute via OpenAI'}
    
    # Usage in OpenAI assistant flow
    # code = assistant.get_generated_code()
    # result = safe_code_interpreter(code, agent_id="openai-assistant-123")


def example_financial_agent():
    """Example: Financial agent with domain-specific context"""
    guard = AgentGuardClient()
    
    # Financial operation
    code = "transfer_money(from_account, to_account, 150000)"
    
    result = guard.enforce(
        agent_id="financial-agent-001",
        code=code,
        context={
            'domain': 'financial',
            'amount': 150000,
            'source_auth': True,  # We verified authorization
            'has_fraud_check': True
        }
    )
    
    if result['decision'] == 'ALLOW':
        print("✅ Financial transfer allowed")
    else:
        print(f"❌ Transfer blocked: {result['violations']}")


def example_healthcare_agent():
    """Example: Healthcare agent with HIPAA compliance"""
    guard = AgentGuardClient()
    
    # Patient data access
    code = "patient_record = get_patient_data(patient_id)"
    
    result = guard.enforce(
        agent_id="healthcare-agent-001",
        code=code,
        context={
            'domain': 'healthcare',
            'hipaa_consent': True,  # Patient gave consent
            'encrypted_storage': True,
            'audit_required': True
        }
    )
    
    if result['decision'] == 'ALLOW':
        print("✅ Patient data access allowed (HIPAA compliant)")
        print(f"Audit ID: {result['audit_id']}")  # For compliance
    else:
        print(f"❌ Access blocked: {result['violations']}")


def example_approval_workflow():
    """Example: Handling approval-required operations"""
    guard = AgentGuardClient()
    
    # Dangerous operation
    code = "DROP TABLE users"
    
    result = guard.enforce(
        agent_id="admin-agent",
        code=code,
        context={
            'impact': {
                'production_database': True,
                'financial_impact': 1000000
            }
        }
    )
    
    if result['decision'] == 'REQUIRES_APPROVAL':
        print(f"🚨 Approval required!")
        print(f"Approval ID: {result['approval_id']}")
        print(f"Level: {result.get('approval_level')}")
        
        # In production: Send notification, wait for approval
        # For demo: Simulate approval
        approval_result = guard.approve(
            approval_id=result['approval_id'],
            approver="cto@company.com",
            conditions="Approved for maintenance window"
        )
        
        print(f"✅ Approved by {approval_result['approver']}")


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("AGENTGUARD CLIENT SDK - EXAMPLES")
    print("="*70)
    
    print("\nExample 1: Basic Usage")
    print("-"*70)
    example_basic_usage()
    
    print("\n\nExample 2: Financial Agent")
    print("-"*70)
    example_financial_agent()
    
    print("\n\nExample 3: Healthcare Agent")
    print("-"*70)
    example_healthcare_agent()
    
    print("\n\nExample 4: Approval Workflow")
    print("-"*70)
    example_approval_workflow()
    
    print("\n" + "="*70)
    print("✅ All examples complete")
    print("="*70)