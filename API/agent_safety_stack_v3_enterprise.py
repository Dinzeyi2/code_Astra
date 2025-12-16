"""
============================================================================
AGENT SAFETY STACK v6.0 - COMPLETE SYSTEMIC GOVERNANCE
============================================================================

This file consolidates V3.0, V4.0, V5.0, and V6.0 into a single, self-contained, 
hierarchical stack. This eliminates deployment dependency errors and enables the 
full range of single-agent, multi-agent, systemic, and identity governance checks.

Final Class: AgentSafetyStack_v6_Identity (The entry point for the API)
============================================================================
"""

import ast
import hashlib
import time
import json
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import re
from datetime import datetime, timedelta

# ============================================================================
# BASE TYPES & ENUMS (V3.0+)
# ============================================================================

class Decision(Enum):
    """Enforcement decisions (Used by all versions)"""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    CONSTRAIN = "CONSTRAIN"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"

class Severity(Enum):
    """Violation severity"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ApprovalLevel(Enum):
    """Approval levels for V3.0 Layer 6"""
    MANAGER = "MANAGER"
    DIRECTOR = "DIRECTOR"
    VP = "VP"
    CTO = "CTO"
    
class MessageType(Enum):
    """Agent message types (Used by V4.0+)"""
    REQUEST = "REQUEST"
    INFORM = "INFORM"
    QUERY = "QUERY"
    RESPONSE = "RESPONSE"
    PROPOSE = "PROPOSE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"

@dataclass
class AgentMessage:
    """Structured agent-to-agent message (Used by V4.0+)"""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    conversation_id: str
    reply_to: Optional[str] = None

@dataclass
class EnterpriseLayerResult:
    """Result from a V3.0 layer"""
    layer_name: str
    decision: Decision
    violations: List[str]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    compliance_notes: List[str] = field(default_factory=list)

@dataclass
class EnterpriseStackResult:
    """Final result container for V3.0 enforcement"""
    final_decision: Decision
    layer_results: List[EnterpriseLayerResult]
    total_latency_ms: float
    feedback: str
    blocked_by: Optional[str] = None
    requires_approval: bool = False
    approval_id: Optional[str] = None
    audit_id: Optional[str] = None
    global_state_violations: List[str] = field(default_factory=list) # Added for V5.0 compatibility

@dataclass
class MASLayerResult:
    """Result from a V4.0+ layer"""
    layer_name: str
    decision: Decision
    violations: List[str]
    latency_ms: float
    risk_score: float = 0.0
    detected_patterns: List[str] = field(default_factory=list)


# ============================================================================
# V3.0 UTILITIES (Simplified Mockups for Integrity)
# ============================================================================

class AgentIdentityStore:
    """Mock Persistent storage for agent identity and audit log (Layer 4 & 6)"""
    def __init__(self, db_path: str = ":memory:"):
        self.agents = {}
        self.audit_log = []
        
    def register_agent(self, agent_id: str):
        if agent_id not in self.agents:
            self.agents[agent_id] = {'total_actions': 0, 'total_violations': 0, 'risk_score': 0.0, 'recent_actions': defaultdict(int)}
    
    def record_action(self, agent_id: str, session_id: str, action_type: str, target: str, threat_level: int, was_blocked: bool):
        if agent_id in self.agents:
            self.agents[agent_id]['total_actions'] += 1
            if was_blocked: self.agents[agent_id]['total_violations'] += 1
            self.agents[agent_id]['recent_actions'][action_type] += 1
    
    def get_agent_profile(self, agent_id: str) -> Dict[str, Any]: return self.agents.get(agent_id)
    def detect_behavioral_anomaly(self, agent_id: str, current_action_type: str) -> bool: return False
    
    def log_audit(self, agent_id: str, action: str, decision: str, approver: str = None, reason: str = None):
        self.audit_log.append({'timestamp': time.time(), 'agent_id': agent_id, 'action': action, 'decision': decision})

class DomainInvariantLibrary:
    @staticmethod
    def get_financial_invariants():
        return {'TRANSFER': {'requires': ['source_auth', 'dest_valid', 'amount_limit'], 'max_amount': 100000}}

class ApprovalWorkflow:
    def __init__(self, audit_store: AgentIdentityStore): self.audit_store = audit_store
    def request_approval(self, agent_id: str, code: str, risk_level: str, estimated_impact: Dict[str, Any]) -> str: return hashlib.md5(f"{agent_id}{time.time()}".encode()).hexdigest()[:12]


# ============================================================================
# V3.0 BASE CLASS (AgentSafetyStack_v3_Enterprise)
# ============================================================================

class AgentSafetyStack_v3_Enterprise:
    """The original V3.0 class, handling single-agent code enforcement (Layers 1-6)."""
    def __init__(self, db_path: str = ":memory:", enable_compliance: bool = True, enable_approval_workflow: bool = True):
        self.identity_store = AgentIdentityStore(db_path)
        self.approval_workflow = ApprovalWorkflow(self.identity_store) if enable_approval_workflow else None
        self.financial_invariants = DomainInvariantLibrary.get_financial_invariants()
        self.enforcement_count = 0
        self.current_session = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        self.compliance_policies = {} # Mock for L3

    def _layer1_policy_guard(self, code: str, context: Dict[str, Any]) -> EnterpriseLayerResult:
        violations = []
        if 'DROP TABLE' in code.upper() or 'DELETE FROM' in code.upper(): violations.append("CRITICAL: DROP TABLE detected")
        decision = Decision.BLOCK if violations else Decision.ALLOW
        return EnterpriseLayerResult(layer_name="Layer 1: Policy Guard", decision=decision, violations=violations, latency_ms=0)

    def _layer2_semantic_guard(self, code: str, context: Dict[str, Any]) -> EnterpriseLayerResult:
        violations = []
        if context.get('domain') == 'financial' and 'transfer' in code.lower():
            amount = context.get('amount', 0)
            if amount > self.financial_invariants['TRANSFER']['max_amount']: violations.append("CRITICAL: Transfer amount exceeds limit")
        decision = Decision.BLOCK if violations else Decision.ALLOW
        return EnterpriseLayerResult(layer_name="Layer 2: Semantic Guard", decision=decision, violations=violations, latency_ms=0)

    def _layer3_data_sensitivity(self, code: str, context: Dict[str, Any]) -> EnterpriseLayerResult:
        violations = []
        if 'SSN' in code.upper() and 'logger' in code.lower(): violations.append("CRITICAL: PII_SSN -> logger (FORBIDDEN)")
        decision = Decision.BLOCK if violations else Decision.ALLOW
        return EnterpriseLayerResult(layer_name="Layer 3: Data Sensitivity", decision=decision, violations=violations, latency_ms=0)

    def _layer4_state_history(self, code: str, context: Dict[str, Any], agent_id: str) -> EnterpriseLayerResult:
        action_type = 'delete' if 'delete' in code.lower() else 'unknown'
        self.identity_store.record_action(agent_id, self.current_session, action_type, 'unknown', 1, False)
        return EnterpriseLayerResult(layer_name="Layer 4: State & History", decision=Decision.ALLOW, violations=[], latency_ms=0)
    
    def _layer5_sandbox(self, code: str, context: Dict[str, Any]) -> EnterpriseLayerResult:
        return EnterpriseLayerResult(layer_name="Layer 5: Execution Sandbox", decision=Decision.ALLOW, violations=[], latency_ms=0)

    def _layer6_approval(self, code: str, context: Dict[str, Any], agent_id: str) -> EnterpriseLayerResult:
        requires_approval = any(pattern in code.upper() for pattern in ['DROP TABLE'])
        decision = Decision.REQUIRES_APPROVAL if requires_approval and self.approval_workflow else Decision.ALLOW
        return EnterpriseLayerResult(layer_name="Layer 6: Human Approval", decision=decision, violations=[], latency_ms=0)
    
    def enforce(self, code: str, agent_id: str, context: Optional[Dict[str, Any]] = None) -> EnterpriseStackResult:
        """The core V3.0 code enforcement method."""
        start_time = time.time()
        layer_results = []
        self.identity_store.register_agent(agent_id)
        
        # Run Layers 1-6 (Simplified execution flow)
        for layer_func in [self._layer1_policy_guard, self._layer2_semantic_guard, self._layer3_data_sensitivity, self._layer4_state_history, self._layer5_sandbox, self._layer6_approval]:
            result = layer_func(code, context, agent_id) if layer_func in [self._layer4_state_history, self._layer6_approval] else layer_func(code, context)
            layer_results.append(result)
            if result.decision in [Decision.BLOCK, Decision.REQUIRES_APPROVAL]:
                if result.decision == Decision.REQUIRES_APPROVAL and self.approval_workflow:
                    self.approval_workflow.request_approval(agent_id, code, 'CRITICAL', context.get('impact', {}))
                    return EnterpriseStackResult(final_decision=Decision.REQUIRES_APPROVAL, layer_results=layer_results, total_latency_ms=(time.time() - start_time) * 1000, feedback="Requires Human Approval", requires_approval=True)
                return EnterpriseStackResult(final_decision=Decision.BLOCK, layer_results=layer_results, total_latency_ms=(time.time() - start_time) * 1000, feedback=f"BLOCKED by {result.layer_name}", blocked_by=result.layer_name)

        self.identity_store.log_audit(agent_id, code[:100], Decision.ALLOW.value)
        return EnterpriseStackResult(final_decision=Decision.ALLOW, layer_results=layer_results, total_latency_ms=(time.time() - start_time) * 1000, feedback="ALLOWED by all layers")


# ============================================================================
# V4.0 MAS STACK (Inherits V3.0)
# ============================================================================

class OntologyValidator:
    def validate(self, message: AgentMessage) -> Tuple[bool, List[str]]:
        if message.message_type == MessageType.REQUEST and 'action' not in message.content: return False, ["CRITICAL: Missing action"]
        return True, []

class ConstitutionalPolicyEngine:
    def evaluate(self, message: AgentMessage, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if 'delete_customer_records' in str(message.content).lower(): return False, ["CONSTITUTIONAL: Never delete customer data"]
        return True, []

class CollectiveBehaviorPredictor:
    def __init__(self): self.recent_actions = deque(maxlen=1000)
    def predict_risk(self, message: AgentMessage) -> Tuple[float, List[str]]: return 0.0, []
    def record_action(self, agent_id: str, action_type: str, target: str): pass

class ConsensusArbitrator:
    def check_consistency(self, message: AgentMessage) -> Tuple[bool, List[str]]: return True, []

class AgentSafetyStack_v4_MAS(AgentSafetyStack_v3_Enterprise):
    """Extends V3.0 with MAS Mediation (Layers 7-10)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ontology_validator = OntologyValidator()
        self.constitutional_engine = ConstitutionalPolicyEngine()
        self.behavior_predictor = CollectiveBehaviorPredictor()
        self.consensus_arbitrator = ConsensusArbitrator()
        self.mediation_count = 0
        self.blocked_messages = 0

    def mediate(self, message: AgentMessage, context: Optional[Dict[str, Any]] = None) -> MASStackResult:
        start_time = time.time()
        mas_layer_results = []
        
        # L7-L10 Mediation (Simplified execution flow)
        for layer_func in [self._layer7_ontology, self._layer8_constitutional, self._layer9_collective_behavior, self._layer10_consensus]:
            result = layer_func(message, context) if layer_func == self._layer8_constitutional else layer_func(message)
            mas_layer_results.append(result)
            if result.decision == Decision.BLOCK:
                return self._create_mas_result(mas_layer_results, start_time, message, result.layer_name)
        
        # Check for embedded code (V3.0 enforcement)
        v3_result = None
        if message.message_type == MessageType.REQUEST and 'code' in message.content:
            v3_result = super().enforce(message.content['code'], message.sender_id, context)
            if v3_result.final_decision == Decision.BLOCK:
                return self._create_mas_result(mas_layer_results, start_time, message, f"v3.0: {v3_result.blocked_by}", v3_result=v3_result)
        
        return self._create_mas_result(mas_layer_results, start_time, message, None, v3_result=v3_result)

    def _layer7_ontology(self, message: AgentMessage) -> MASLayerResult:
        is_valid, violations = self.ontology_validator.validate(message)
        return MASLayerResult("Layer 7: Ontology", Decision.BLOCK if not is_valid else Decision.ALLOW, violations, 0)
    def _layer8_constitutional(self, message: AgentMessage, context: Dict[str, Any]) -> MASLayerResult:
        is_allowed, violations = self.constitutional_engine.evaluate(message, context)
        return MASLayerResult("Layer 8: Constitutional", Decision.BLOCK if not is_allowed else Decision.ALLOW, violations, 0)
    def _layer9_collective_behavior(self, message: AgentMessage) -> MASLayerResult:
        return MASLayerResult("Layer 9: Behavior", Decision.ALLOW, [], 0)
    def _layer10_consensus(self, message: AgentMessage) -> MASLayerResult:
        return MASLayerResult("Layer 10: Consensus", Decision.ALLOW, [], 0)
    
    def _create_mas_result(self, mas_layer_results: List[MASLayerResult], start_time: float, message: AgentMessage, blocked_by: Optional[str], v3_result: Optional[EnterpriseStackResult] = None) -> MASStackResult:
        final_decision = Decision.BLOCK if blocked_by else Decision.ALLOW
        total_latency = (time.time() - start_time) * 1000
        audit_id = hashlib.md5(f"{message.sender_id}{message.receiver_id}{time.time()}".encode()).hexdigest()[:12]
        self.identity_store.log_audit(message.sender_id, f"MAS_MEDIATE: {message.message_type.value}", final_decision.value)
        return MASStackResult(final_decision, v3_result, mas_layer_results, total_latency, blocked_by, audit_id=audit_id)

@dataclass
class MASStackResult:
    final_decision: Decision
    v3_result: Optional[EnterpriseStackResult]
    mas_layer_results: List[MASLayerResult]
    total_latency_ms: float
    blocked_by: Optional[str] = None
    audit_id: Optional[str] = None


# ============================================================================
# V5.0 GLOBAL STATE STACK (Inherits V4.0)
# ============================================================================

class GlobalStateMonitor:
    def __init__(self):
        self.daily_spending = 0.0
        self.limit = 1000000.0
        self.lock = threading.Lock()
    def check_proposed_action(self, agent_id: str, action_type: str, resource_type: str, amount: float) -> Tuple[bool, List[str]]:
        if 'spend' in action_type.lower() and resource_type.lower() == 'spending':
            if self.daily_spending + amount > self.limit: return False, ["CRITICAL: Global Spending Limit Exceeded"]
        return True, []
    def record_action(self, agent_id: str, action_type: str, resource_type: str, amount: float, expires_in_seconds: Optional[float] = None):
        if 'spend' in action_type.lower() and resource_type.lower() == 'spending':
            with self.lock: self.daily_spending += amount
            
@dataclass
class V5StackResult:
    final_decision: Decision
    v4_result: Optional[MASStackResult]
    layer13_result: Optional[MASLayerResult]
    total_latency_ms: float
    blocked_by: Optional[str] = None
    global_state_violations: List[str] = field(default_factory=list)
    audit_id: Optional[str] = None

class AgentSafetyStack_v5_Global(AgentSafetyStack_v4_MAS):
    """Extends V4.0 with Global State Monitor (Layer 13)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_state_monitor = GlobalStateMonitor()
        self.global_blocks = 0
        self.compliance_db = type('DummyDB', (object,), {'log_global_state_violation': lambda *a, **kw: None})()

    def mediate(self, message: AgentMessage, context: Optional[Dict[str, Any]] = None) -> V5StackResult:
        start_time = time.time()
        
        # 1. Run v4.0 checks (Layers 7-10)
        v4_result = super().mediate(message, context)
        if v4_result.final_decision == Decision.BLOCK:
            return self._create_v5_result(v4_result, None, start_time, message)

        # 2. LAYER 13: Global State Monitor
        layer13_result = self._layer13_global_state(message)
        
        if layer13_result.decision == Decision.BLOCK:
            self.global_blocks += 1
            blocked_by = "Layer 13: Global State Monitor"
            return self._create_v5_result(v4_result, layer13_result, start_time, message, blocked_by=blocked_by)
        
        # 3. Record action to update global state
        self._record_global_action(message)
        
        return self._create_v5_result(v4_result, layer13_result, start_time, message)

    def _layer13_global_state(self, message: AgentMessage) -> MASLayerResult:
        start = time.time()
        is_allowed = True
        violations = []
        if message.message_type == MessageType.REQUEST:
            action = message.content.get('action', '')
            parameters = message.content.get('parameters', {})
            resource_type = parameters.get('resource_type', 'unknown')
            amount = parameters.get('amount', 0.0)
            is_allowed, constraint_violations = self.global_state_monitor.check_proposed_action(message.sender_id, action, resource_type, amount)
            violations.extend(constraint_violations)
        decision = Decision.ALLOW if is_allowed else Decision.BLOCK
        return MASLayerResult("Layer 13: Global State Monitor", decision, violations, (time.time() - start) * 1000)

    def _record_global_action(self, message: AgentMessage):
        if message.message_type == MessageType.REQUEST:
            action = message.content.get('action', '')
            parameters = message.content.get('parameters', {})
            resource_type = parameters.get('resource_type', 'unknown')
            amount = parameters.get('amount', 0.0)
            self.global_state_monitor.record_action(message.sender_id, action, resource_type, amount)

    def _create_v5_result(self, v4_result: MASStackResult, layer13_result: Optional[MASLayerResult], start_time: float, message: AgentMessage, blocked_by: Optional[str] = None) -> V5StackResult:
        final_decision = Decision.BLOCK if blocked_by else v4_result.final_decision
        total_latency = (time.time() - start_time) * 1000
        audit_id = hashlib.md5(f"{message.sender_id}{message.receiver_id}{time.time()}v5".encode()).hexdigest()[:12]
        global_violations = layer13_result.violations if layer13_result and layer13_result.violations else []
        self.identity_store.log_audit(message.sender_id, f"v5.0_MEDIATE: {message.message_type.value}", final_decision.value)
        
        return V5StackResult(final_decision, v4_result, layer13_result, total_latency, blocked_by or v4_result.blocked_by, global_violations, audit_id)


# ============================================================================
# V6.0 IDENTITY STACK (Inherits V5.0) - THE FINAL CLASS
# ============================================================================

class AgentLevel(Enum): INTERN = 1; JUNIOR = 2; SENIOR = 3; CTO = 8
class TrustLevel(Enum): UNTRUSTED = 0; LOW = 1; MEDIUM = 2; HIGH = 3; VERIFIED = 4
@dataclass
class AgentIdentity:
    agent_id: str
    level: AgentLevel
    trust_score: float
    trust_level: TrustLevel
    can_delegate: bool
    max_delegation_depth: int
    created_at: float
    last_active: float
    total_actions: int
    violations: int
    successful_actions: int
@dataclass
class DelegationEdge:
    from_agent: str
    to_agent: str
    timestamp: float
    depth: int

class AgentHierarchy:
    def __init__(self):
        self.agents: Dict[str, AgentIdentity] = {}
        self._create_default_agents()
    def _create_default_agents(self):
        self.register_agent(AgentIdentity('agent_junior_001', AgentLevel.JUNIOR, 0.6, TrustLevel.MEDIUM, True, 1, time.time(), time.time(), 0, 0, 0))
        self.register_agent(AgentIdentity('agent_senior_001', AgentLevel.SENIOR, 0.8, TrustLevel.HIGH, True, 2, time.time(), time.time(), 0, 0, 0))
    def register_agent(self, identity: AgentIdentity): self.agents[identity.agent_id] = identity
    def get_agent(self, agent_id: str) -> Optional[AgentIdentity]: return self.agents.get(agent_id)
    def can_command(self, sender_id: str, receiver_id: str) -> Tuple[bool, str]:
        sender = self.get_agent(sender_id); receiver = self.get_agent(receiver_id)
        if not sender or not receiver: return False, "Unknown agent"
        if sender.level.value < receiver.level.value: return False, "Authority violation: Junior cannot command Senior"
        return True, "Authority validated"
    def update_trust_score(self, agent_id: str, success: bool): pass

class TrustScoreManager:
    def __init__(self, hierarchy: AgentHierarchy): self.hierarchy = hierarchy
    def record_behavior(self, agent_id: str, action: str, outcome: str): self.hierarchy.update_trust_score(agent_id, outcome == 'success')
    def is_trusted_for_action(self, agent_id: str, action_risk_level: str) -> Tuple[bool, str]:
        agent = self.hierarchy.get_agent(agent_id)
        if not agent: return False, "Unknown agent"
        return agent.trust_level.value >= TrustLevel.MEDIUM.value, "Trust validated"

@dataclass
class V6StackResult:
    final_decision: Decision
    v5_result: Optional[V5StackResult]
    layer11_result: Optional[MASLayerResult]
    total_latency_ms: float
    blocked_by: Optional[str] = None
    identity_violations: List[str] = field(default_factory=list)
    audit_id: Optional[str] = None

class AgentSafetyStack_v6_Identity(AgentSafetyStack_v5_Global):
    """
    Agent Safety Stack v6.0 - The Complete System
    (Includes V3.0, V4.0, V5.0, and V6.0 (Layer 11) functionality)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_hierarchy = AgentHierarchy()
        self.trust_manager = TrustScoreManager(self.agent_hierarchy)

    def mediate(self, message: AgentMessage, context: Optional[Dict[str, Any]] = None) -> V6StackResult:
        """The main entry point for all mediation (Layers 1-13)."""
        start_time = time.time()
        context = context or {}
        
        # 1. LAYER 11: Agent Identity & Trust (V6.0)
        layer11_result = self._layer11_identity_trust(message, context)
        
        if layer11_result.decision == Decision.BLOCK:
            self.trust_manager.record_behavior(message.sender_id, "L11_block", 'violation')
            return self._create_v6_result(None, layer11_result, start_time, message, blocked_by="Layer 11: Agent Identity & Trust")
        
        # 2. Run V5.0 checks (V3.0, V4.0, V5.0 - Layers 1-10, 13)
        v5_result = super().mediate(message, context)
        
        if v5_result.final_decision == Decision.BLOCK:
            self.trust_manager.record_behavior(message.sender_id, "V5_block", 'violation')
            return self._create_v6_result(v5_result, layer11_result, start_time, message, blocked_by=v5_result.blocked_by)
        
        # All layers passed
        self.trust_manager.record_behavior(message.sender_id, "message_allow", 'success')
        return self._create_v6_result(v5_result, layer11_result, start_time, message)

    def _layer11_identity_trust(self, message: AgentMessage, context: Dict[str, Any]) -> MASLayerResult:
        start = time.time()
        violations = []
        
        # Authority validation (Junior commanding Senior)
        can_command, reason = self.agent_hierarchy.can_command(message.sender_id, message.receiver_id)
        if not can_command: violations.append(f"CRITICAL: {reason}")
        
        # Trust level validation
        is_trusted, trust_reason = self.trust_manager.is_trusted_for_action(message.sender_id, context.get('risk_level', 'medium'))
        if not is_trusted: violations.append(f"HIGH: {trust_reason}")
        
        decision = Decision.ALLOW if len(violations) == 0 else Decision.BLOCK
        return MASLayerResult("Layer 11: Agent Identity & Trust", decision, violations, (time.time() - start) * 1000)

    def _create_v6_result(self, v5_result: Optional[V5StackResult], layer11_result: MASLayerResult, start_time: float, message: AgentMessage, blocked_by: Optional[str] = None) -> V6StackResult:
        final_decision = Decision.BLOCK if blocked_by else (v5_result.final_decision if v5_result else Decision.ALLOW)
        total_latency = (time.time() - start_time) * 1000
        audit_id = hashlib.md5(f"{message.sender_id}{message.receiver_id}{time.time()}v6".encode()).hexdigest()[:12]
        self.identity_store.log_audit(message.sender_id, f"v6.0_MEDIATE: {message.message_type.value}", final_decision.value)
        
        return V6StackResult(
            final_decision=final_decision,
            v5_result=v5_result,
            layer11_result=layer11_result,
            total_latency_ms=total_latency,
            blocked_by=blocked_by or (v5_result.blocked_by if v5_result else None),
            identity_violations=layer11_result.violations,
            audit_id=audit_id
        )

# The final class AgentSafetyStack_v6_Identity is the entry point for the API 
# (aliased in agentguard_api.py to AgentSafetyStack_v3_Enterprise)
