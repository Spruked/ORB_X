"""
UCM ORB - Unified Cognitive Matrix Orbital Carrier
====================================================

ARCHITECTURAL ONTOLOGY (Immutable)
----------------------------------

AXIOMS:
1. Sovereignty without supremacy
2. Geometry over hierarchy  
3. Influence without authority
4. Truth must be earned before it can shortcut
5. Humans interact with exactly one mind
6. Execution must never become cognition
7. Self-improvement is proposed, never self-authorized
8. Continuity is a cognitive function, not a data structure

CARRIER LAW:
CALI may adapt her body, but she never amputates her mind.

COMPONENTS:
- Space Field: Semantic manifold where reasoning occurs
- Ghost Trace Layer: Decaying influence, never authority
- Core-4: Four sovereign, independent traversers
- SoftMax Advisor: Statistical synthesis, never decision
- Convergence Core: Event where coherence emerges
- CALI: Frontal lobe, articulator, continuity holder
- Vaults: Apriori (foundational) and Posteriori (resolved) truth storage
- Worker Swarm: Execution-only, no reasoning
- Forges: Propose improvements, require CALI approval

GOVERNANCE RULES (v1 Doctrine):
A. Ghost Trace Decay: hybrid (time + reinforcement + confidence)
B. Convergence Threshold: ≥3/4, SoftMax plateau, abstention allowed, ECM block possible
C. Vault Ingress: Posteriori default, promotion to Apriori requires N uses + T time + ECM no-veto + CALI stability mark
D. Worker Escalation: Novelty + value + risk routing; 2% human escalation target
E. Forge Authority: Proposal cycle, 3-strike rule, CALI override capability

This is the spine. Everything else is body.
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio

# ============================================================================
# 0. DOCTRINE VALIDATOR
# ============================================================================


class DoctrineViolation(Exception):
    def __init__(self, stage: str, violations: List[str]):
        super().__init__("; ".join(violations))
        self.stage = stage
        self.violations = violations


class Doctrine:
    """Runtime guardrail for Architectural Ontology."""

    def __init__(self):
        self.axioms = [
            "sovereignty_without_supremacy",
            "geometry_over_hierarchy",
            "influence_without_authority",
            "earned_truth_shortcuts",
            "single_human_mind",
            "execution_never_cognition",
            "propose_not_self_authorize",
            "continuity_is_cognitive",
        ]

        self.governance = {
            "ghost_trace_decay": "hybrid",
            "convergence_threshold": "3_of_4",
            "vault_ingress": "posteriori_default",
            "worker_escalation": "novelty_value_risk",
            "forge_authority": "proposal_cycle",
        }

        self.events: List[Dict[str, Any]] = []

    def spine_check(self, stage: str, **kwargs) -> Dict[str, Any]:
        """Validate a stage against doctrine; returns {ok, violations}."""
        violations: List[str] = []

        if stage == "routing":
            target = kwargs.get("target")
            query = kwargs.get("query", "")
            if target == "worker" and self._is_cognitive_query(query):
                violations.append("Execution boundary: cognitive query routed to worker")

        if stage == "convergence":
            softmax = kwargs.get("softmax", {})
            if softmax:
                max_weight = max(softmax.values())
                if max_weight > 0.75 and len(softmax) >= 3:
                    violations.append("Sovereignty check: single core dominance detected")

        if stage == "articulation":
            route = kwargs.get("route")
            result = kwargs.get("result", {})
            if route == "worker" and result.get("status") == "converged":
                violations.append("Execution boundary: worker output attempted convergence claims")

        ok = len(violations) == 0
        if not ok:
            self.events.append({"stage": stage, "violations": violations, "timestamp": time.time()})

        return {"ok": ok, "stage": stage, "violations": violations}

    def _is_cognitive_query(self, query: str) -> bool:
        cognitive_markers = ["why", "meaning", "interpret", "theory", "design", "intent"]
        query_lower = query.lower()
        return any(marker in query_lower for marker in cognitive_markers)

# ============================================================================
# I. SPACE FIELD & GHOST TRACE LAYER
# ============================================================================

@dataclass
class ConceptAttractor:
    id: str
    position: Tuple[float, float, float]
    mass: float
    category: str
    confidence: float = 0.5

@dataclass
class GhostTrace:
    source_core: str
    position: Tuple[float, float, float]
    confidence: float
    timestamp: float
    trace_type: str  # "empirical", "pattern", "categorical", "coherence"
    
    def decay_factor(self, current_time: float) -> float:
        """Hybrid decay: time + confidence"""
        age = current_time - self.timestamp
        base_decay = 0.5 ** (age / 3600)  # 1-hour half-life base
        confidence_boost = 1 + (self.confidence * 0.5)
        return base_decay * confidence_boost

class SpaceField:
    """Semantic manifold where reasoning occurs. SF never decides, only permits movement."""
    
    def __init__(self):
        self.attractors: Dict[str, ConceptAttractor] = {}
        self.ghost_traces: List[GhostTrace] = []
        self.tension_map: Dict[str, float] = {}  # region -> tension level
        
    def add_attractor(self, attractor: ConceptAttractor):
        self.attractors[attractor.id] = attractor
        
    def add_ghost_trace(self, trace: GhostTrace):
        self.ghost_traces.append(trace)
        # Clean old traces periodically
        self._prune_traces(trace.timestamp)
        
    def _prune_traces(self, current_time: float):
        """Remove fully decayed traces"""
        self.ghost_traces = [
            t for t in self.ghost_traces 
            if t.decay_factor(current_time) > 0.01
        ]
        
    def get_influence_at(self, position: Tuple[float, float, float], current_time: float) -> Dict[str, float]:
        """Calculate total ghost influence at a position"""
        influence = {"empirical": 0.0, "pattern": 0.0, "categorical": 0.0, "coherence": 0.0}
        
        for trace in self.ghost_traces:
            distance = self._distance(position, trace.position)
            if distance < 10.0:  # Influence radius
                decayed_confidence = trace.confidence * trace.decay_factor(current_time)
                influence[trace.trace_type] += decayed_confidence / (distance + 1)
                
        return influence
    
    def _distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2) ** 0.5

# ============================================================================
# II. CORE-4 SOVEREIGN MINDS
# ============================================================================

@dataclass
class TraversalVerdict:
    core_id: str
    verdict: str
    confidence: float
    position: Tuple[float, float, float]
    traversal_modes: List[str]
    timestamp: float = field(default_factory=time.time)
    
class CoreMind:
    """Base class for Core-4. Each is sovereign and independent."""
    
    def __init__(self, core_id: str, specialization: str):
        self.core_id = core_id
        self.specialization = specialization
        self.traversal_modes = []
        
    def traverse(self, sf: SpaceField, query: str) -> TraversalVerdict:
        """Sovereign traversal of Space Field. No visibility into other minds."""
        # Each core implements its own traversal logic
        raise NotImplementedError
        
class CaleonGenesis(CoreMind):
    """Embodiment, identity, narrative coherence"""
    
    def __init__(self):
        super().__init__("caleon", "Embodiment & Continuity")
        self.traversal_modes = ["coherence", "empirical", "categorical"]
        
    def traverse(self, sf: SpaceField, query: str) -> TraversalVerdict:
        # Embodiment-focused traversal
        position = (0.0, 1.0, 0.0)  # Narrative/corpo-real axis
        influence = sf.get_influence_at(position, time.time())
        
        # Prefer coherence paths
        confidence = 0.7 + (influence["coherence"] * 0.3)
        
        return TraversalVerdict(
            core_id=self.core_id,
            verdict=f"Embodied continuity: {query} aligns with identity trajectory",
            confidence=min(confidence, 1.0),
            position=position,
            traversal_modes=self.traversal_modes
        )

class CaliXOne(CoreMind):
    """Structural intelligence, pattern architecture, elegant form"""
    
    def __init__(self):
        super().__init__("cali_x", "Structure & Pattern")
        self.traversal_modes = ["pattern", "categorical", "coherence"]
        
    def traverse(self, sf: SpaceField, query: str) -> TraversalVerdict:
        # Structure-focused traversal
        position = (1.0, 0.0, 0.0)  # Formal/pattern axis
        
        # Detect pattern resonance
        influence = sf.get_influence_at(position, time.time())
        confidence = 0.8 + (influence["pattern"] * 0.2)
        
        return TraversalVerdict(
            core_id=self.core_id,
            verdict=f"Structural resolution: {query} fits established pattern architecture",
            confidence=min(confidence, 1.0),
            position=position,
            traversal_modes=self.traversal_modes
        )

class KayGeeOne(CoreMind):
    """Exploratory, harmonic, novelty-seeking"""
    
    def __init__(self):
        super().__init__("kaygee", "Exploration & Harmony")
        self.traversal_modes = ["pattern", "coherence", "empirical"]
        
    def traverse(self, sf: SpaceField, query: str) -> TraversalVerdict:
        # Exploration-focused traversal
        position = (-1.0, 0.5, 0.5)  # Novelty/harmony axis
        
        # Seek unexplored regions with low ghost influence
        influence = sf.get_influence_at(position, time.time())
        total_influence = sum(influence.values())
        
        # Lower influence = higher confidence (unexplored territory)
        confidence = 0.6 + ((10.0 - total_influence) / 10.0 * 0.4)
        
        return TraversalVerdict(
            core_id=self.core_id,
            verdict=f"Novel harmonic inference: {query} reveals new pattern alignment",
            confidence=min(max(confidence, 0.3), 1.0),
            position=position,
            traversal_modes=self.traversal_modes
        )

class ECM(CoreMind):
    """Executive Convergence Mind - adjudication, authority, governance"""
    
    def __init__(self):
        super().__init__("ecm", "Adjudication & Governance")
        self.traversal_modes = ["categorical", "empirical", "coherence"]
        
    def traverse(self, sf: SpaceField, query: str) -> TraversalVerdict:
        # Authority-focused traversal
        position = (0.0, 0.0, 1.0)  # Governance/structural integrity axis
        
        # Assess overall field tension
        influence = sf.get_influence_at(position, time.time())
        tension = sf.tension_map.get("global", 0.5)
        
        # High tension = lower confidence (caution)
        confidence = 0.9 - (tension * 0.2)
        
        return TraversalVerdict(
            core_id=self.core_id,
            verdict=f"Epistemic authority: {query} converges to stable truth under governance",
            confidence=min(confidence, 1.0),
            position=position,
            traversal_modes=self.traversal_modes
        )

# ============================================================================
# III. SOFTMAX ADVISOR & CONVERGENCE CORE
# ============================================================================

class SoftMaxAdvisor:
    """Statistical synthesizer. Reports weather, never flies plane."""
    
    def synthesize(self, verdicts: List[TraversalVerdict]) -> Dict[str, float]:
        """Produce confidence-weighted probability distribution"""
        if not verdicts:
            return {}
            
        total_confidence = sum(v.confidence for v in verdicts)
        if total_confidence == 0:
            return {v.core_id: 1.0/len(verdicts) for v in verdicts}
            
        return {
            v.core_id: v.confidence / total_confidence 
            for v in verdicts
        }

class ConvergenceCore:
    """Event where coherence emerges. Not invoked; it happens."""
    
    def __init__(self):
        self.convergence_history: List[Dict] = []
        
    def should_converge(self, verdicts: List[TraversalVerdict]) -> bool:
        """Governance Rule B: ≥3/4 returns + plateau"""
        if len(verdicts) < 3:
            return False
            
        # SoftMax plateau check (standard deviation < threshold)
        if len(verdicts) >= 4:
            confidences = [v.confidence for v in verdicts]
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            if variance > 0.05:  # Threshold for "plateau"
                return False
                
        return True
        
    def converge(self, verdicts: List[TraversalVerdict], softmax_dist: Dict[str, float]) -> Dict[str, Any]:
        """Governance Rule B: Accept abstention, allow ECM block"""
        
        # Check for ECM block
        ecm_verdict = next((v for v in verdicts if v.core_id == "ecm"), None)
        if ecm_verdict and ecm_verdict.confidence < 0.5:
            return {
                "status": "blocked",
                "reason": "ECM integrity flag raised",
                "ecm_verdict": ecm_verdict
            }
        
        # Weighted consensus
        weighted_positions = [0.0, 0.0, 0.0]
        total_weight = 0.0
        
        for verdict in verdicts:
            weight = softmax_dist.get(verdict.core_id, 0)
            weighted_positions[0] += verdict.position[0] * weight
            weighted_positions[1] += verdict.position[1] * weight
            weighted_positions[2] += verdict.position[2] * weight
            total_weight += weight
            
        if total_weight == 0:
            return {
                "status": "failed",
                "reason": "Zero convergence weight"
            }
            
        final_position = tuple(wp / total_weight for wp in weighted_positions)
        
        result = {
            "status": "converged",
            "position": final_position,
            "confidence": sum(v.confidence for v in verdicts) / len(verdicts),
            "contributing_minds": [v.core_id for v in verdicts],
            "ecm_authority": ecm_verdict.confidence if ecm_verdict else 0.0
        }
        
        # Log convergence event
        self.convergence_history.append({
            "timestamp": time.time(),
            "result": result
        })
        
        return result

# ============================================================================
# IV. VAULT SYSTEM (APRIORI / POSTERIORI)
# ============================================================================

class VaultSystem:
    """Epistemic muscle memory. Truth accelerates only after earning."""
    
    def __init__(self, vault_dir: Path):
        self.vault_dir = vault_dir
        self.apriori_path = vault_dir / "apriori.jsonl"
        self.posteriori_path = vault_dir / "posteriori.jsonl"
        
        vault_dir.mkdir(parents=True, exist_ok=True)
        
        # Governance Rule C parameters
        self.N_USES_THRESHOLD = 5  # For Apriori promotion
        self.T_TIME_THRESHOLD = 86400  # 24 hours in seconds
        
    def add_posteriori(self, inference: Dict[str, Any], source: str):
        """Add resolved inference to posteriori"""
        entry = {
            "timestamp": time.time(),
            "inference": inference,
            "source": source,
            "uses": 0,
            "last_used": time.time()
        }
        
        with open(self.posteriori_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
            
    def try_apriori_bypass(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Check if query is known via Apriori"""
        if not self.apriori_path.exists():
            return None
            
        with open(self.apriori_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("query_hash") == query_hash:
                    # Found apriori truth - instant bypass
                    return entry.get("inference")
                    
        return None
        
    def promote_to_apriori(self, posteriori_entry: Dict[str, Any]):
        """Governance Rule C: Promote based on N uses + T time + ECM no-veto"""
        current_time = time.time()
        age = current_time - posteriori_entry["timestamp"]
        
        if (posteriori_entry["uses"] >= self.N_USES_THRESHOLD and 
            age >= self.T_TIME_THRESHOLD):
            
            # Create apriori entry
            apriori_entry = {
                "timestamp": current_time,
                "query_hash": self._hash_query(posteriori_entry["inference"]),
                "inference": posteriori_entry["inference"],
                "source": posteriori_entry["source"],
                "promoted_at": current_time
            }
            
            with open(self.apriori_path, 'a') as f:
                f.write(json.dumps(apriori_entry) + "\n")
                
    def _hash_query(self, inference: Dict[str, Any]) -> str:
        """Create stable hash for inference"""
        content = json.dumps(inference, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

# ============================================================================
# V. WORKER SWARM & FORGES
# ============================================================================

class WorkerSwarm:
    """Execution-only swarm. Workers do, never understand why."""
    
    def __init__(self):
        self.worker_registry: Dict[str, Dict] = {}
        self.escalation_count = 0
        
    def register_worker(self, worker_id: str, domain: str, logic_path: Path):
        self.worker_registry[worker_id] = {
            "domain": domain,
            "logic_path": logic_path,
            "call_count": 0,
            "success_rate": 1.0
        }
        
    def execute(self, worker_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute worker task. Returns result or escalates."""
        
        if worker_id not in self.worker_registry:
            return self._escalate("worker_not_found", task)
            
        worker = self.worker_registry[worker_id]
        worker["call_count"] += 1
        
        # Governance Rule D: Check for novelty/value/risk
        if self._is_novel(task) or self._is_high_value(task):
            return self._escalate("novelty_high_value", task, worker_id)
            
        # Simulate worker execution (in reality, calls worker process)
        success = worker["success_rate"] > 0.7
        
        if success:
            result = {
                "status": "success",
                "worker_id": worker_id,
                "result": f"Task {task['id']} completed by {worker_id}",
                "domain": worker["domain"]
            }
        else:
            result = self._escalate("worker_failure", task, worker_id)
            
        return result
        
    def _is_novel(self, task: Dict[str, Any]) -> bool:
        """Governance Rule D: Novelty detection"""
        # Simple heuristic: tasks without prior pattern
        return task.get("novelty_score", 0) > 0.7
        
    def _is_high_value(self, task: Dict[str, Any]) -> bool:
        """Governance Rule D: Value-based routing"""
        return task.get("value_score", 0) > 0.8
        
    def _escalate(self, reason: str, task: Dict[str, Any], worker_id: str = None) -> Dict[str, Any]:
        """Escalate to CALI, not Core-4 directly"""
        self.escalation_count += 1
        
        return {
            "status": "escalated",
            "reason": reason,
            "task": task,
            "worker_id": worker_id,
            "escalation_id": f"esc_{int(time.time())}_{self.escalation_count}"
        }

class ForgeSystem:
    """Restricted self-improvement. Forges propose, CALI guards doctrine."""
    
    def __init__(self, forge_dir: Path):
        self.forge_dir = forge_dir
        forge_dir.mkdir(parents=True, exist_ok=True)
        
        self.pending_proposals: List[Dict] = []
        
        # Governance Rule E parameters
        self.MAX_PROPOSAL_TRIES = 3
        
    def propose_improvement(self, forge_type: str, proposal: Dict[str, Any]) -> str:
        """Forge proposes improvement. Returns proposal ID."""
        
        proposal_id = f"prop_{forge_type}_{int(time.time())}"
        
        proposal_entry = {
            "id": proposal_id,
            "forge_type": forge_type,
            "proposal": proposal,
            "timestamp": time.time(),
            "status": "pending",
            "attempts": 0
        }
        
        self.pending_proposals.append(proposal_entry)
        
        return proposal_id
        
    def approve_proposal(self, proposal_id: str, approved: bool, cali_notes: str = ""):
        """CALI approves or rejects proposal"""
        
        for proposal in self.pending_proposals:
            if proposal["id"] == proposal_id:
                if approved:
                    proposal["status"] = "approved"
                    proposal["cali_notes"] = cali_notes
                else:
                    proposal["attempts"] += 1
                    if proposal["attempts"] >= self.MAX_PROPOSAL_TRIES:
                        proposal["status"] = "rejected_final"
                        # Governance Rule E: Escalate to ECM
                        self._escalate_to_ecm(proposal)
                    else:
                        proposal["status"] = "rejected_retry"
                        proposal["cali_notes"] = cali_notes
                        
                return
                
    def _escalate_to_ecm(self, proposal: Dict[str, Any]):
        """After 3 rejections, escalate to ECM for structural review"""
        """This would notify ECM mind to review proposal
        For now, it's a placeholder operation."""
        return

# ============================================================================
# VI. CALI - THE CARRIER & FRONTAL LOBE
# ============================================================================

class CALI:
    """
    The only human-accessible mind.
    Carrier of UCM ontology.
    Frontal lobe: articulator, continuity holder, first resolver.
    """
    
    def __init__(self, orb_dir: Path):
        self.orb_dir = orb_dir
        self.space_field = SpaceField()
        self.core_minds = {
            "caleon": CaleonGenesis(),
            "cali_x": CaliXOne(),
            "kaygee": KayGeeOne(),
            "ecm": ECM()
        }
        self.softmax = SoftMaxAdvisor()
        self.convergence = ConvergenceCore()
        self.vaults = VaultSystem(orb_dir / "vaults")
        self.workers = WorkerSwarm()
        self.forges = ForgeSystem(orb_dir / "forges")
        self.doctrine = Doctrine()
        
        # Load initial attractors
        self._initialize_space_field()
        
        # Interaction state
        self.session_continuity: Dict[str, Any] = {}
        self.health_status = "healthy"
        
    def _initialize_space_field(self):
        """Seed Space Field with fundamental concepts"""
        concepts = [
            ("identity", (0.0, 1.0, 0.0), "continuity", 0.9),
            ("structure", (1.0, 0.0, 0.0), "form", 0.85),
            ("exploration", (-1.0, 0.5, 0.5), "novelty", 0.7),
            ("governance", (0.0, 0.0, 1.0), "authority", 0.95)
        ]
        
        for cid, pos, cat, conf in concepts:
            self.space_field.add_attractor(
                ConceptAttractor(id=cid, position=pos, mass=1.0, category=cat, confidence=conf)
            )
            
    async def process_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        CALI's primary interface.
        Carrier Law: Never surrender doctrine, always adapt body.
        """
        
        # 1. Check Apriori Vault (instant bypass)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        apriori_result = self.vaults.try_apriori_bypass(query_hash)
        if apriori_result:
            return {
                "status": "apriori_bypass",
                "result": apriori_result,
                "source": "vault",
                "cali_continuity": self.session_continuity.get(user_id, {})
            }
            
        # 2. Route to appropriate subsystem
        routing_decision = self._route_query(query)

        # Doctrine: prevent execution from handling cognition-heavy queries
        doctrine_gate = self.doctrine.spine_check("routing", query=query, target=routing_decision["target"])
        if not doctrine_gate["ok"]:
            routing_decision = {"target": "cognitive", "doctrine_override": doctrine_gate["violations"]}
        
        if routing_decision["target"] == "worker":
            # Procedural task - delegate to worker
            worker_result = self.workers.execute(
                routing_decision["worker_id"],
                {"id": f"task_{int(time.time())}", "query": query, **routing_decision}
            )
            
            if worker_result["status"] == "escalated":
                # Worker escalated to CALI, now route to Core-4
                return await self._cognitive_pipeline(query, user_id, escalation_context=worker_result)
            else:
                # Update Posteriori Vault
                self.vaults.add_posteriori(worker_result, f"worker_{routing_decision['worker_id']}")

                doctrine_art = self.doctrine.spine_check("articulation", route="worker", result=worker_result)
                if not doctrine_art["ok"]:
                    return self._articulate_block({"reason": "; ".join(doctrine_art["violations"])}, user_id)

                return self._articulate_result(worker_result, user_id)
                
        elif routing_decision["target"] == "cognitive":
            # Cognitive task - Core-4 pipeline
            return await self._cognitive_pipeline(query, user_id)
            
        else:
            # Direct resolution path
            return self._articulate_result(routing_decision["result"], user_id)
            
    def _route_query(self, query: str) -> Dict[str, Any]:
        """Simple routing logic - can be enhanced by Worker Forge"""
        # Heuristic: if query contains "how to", "process", "step", route to worker
        procedural_keywords = ["how to", "process", "step", "execute", "run"]
        if any(kw in query.lower() for kw in procedural_keywords):
            return {"target": "worker", "worker_id": "default_process_worker"}
            
        # Default to cognitive pipeline
        return {"target": "cognitive"}
        
    async def _cognitive_pipeline(self, query: str, user_id: str, escalation_context: Dict = None) -> Dict[str, Any]:
        """Full Core-4 traversal, SoftMax synthesis, Convergence, CALI articulation"""
        
        # 1. Core-4 independently traverse Space Field
        verdicts = []
        for core in self.core_minds.values():
            verdict = core.traverse(self.space_field, query)
            verdicts.append(verdict)
            
            # Add ghost trace
            trace = GhostTrace(
                source_core=core.core_id,
                position=verdict.position,
                confidence=verdict.confidence,
                timestamp=verdict.timestamp,
                trace_type=verdict.traversal_modes[0]  # Primary mode
            )
            self.space_field.add_ghost_trace(trace)
            
        # 2. SoftMax synthesizes
        softmax_dist = self.softmax.synthesize(verdicts)

        doctrine_conv = self.doctrine.spine_check("convergence", verdicts=verdicts, softmax=softmax_dist)
        if not doctrine_conv["ok"]:
            return self._articulate_block({"reason": "; ".join(doctrine_conv["violations"])}, user_id)
        
        # 3. Convergence (if conditions met)
        if self.convergence.should_converge(verdicts):
            convergence_result = self.convergence.converge(verdicts, softmax_dist)
            
            if convergence_result["status"] == "blocked":
                # ECM blocked - cannot proceed
                return self._articulate_block(convergence_result, user_id)
                
            elif convergence_result["status"] == "converged":
                # Success - store in Posteriori Vault
                self.vaults.add_posteriori(convergence_result, "convergence_core")
                
                doctrine_art = self.doctrine.spine_check("articulation", route="cognitive", result=convergence_result)
                if not doctrine_art["ok"]:
                    return self._articulate_block({"reason": "; ".join(doctrine_art["violations"])}, user_id)

                # Articulate to user
                return self._articulate_convergence(convergence_result, user_id, verdicts)
        else:
            # Not enough consensus - CALI decides whether to wait or escalate
            return self._articulate_pending(verdicts, softmax_dist, user_id)
            
    def _articulate_result(self, result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """CALI's narrative articulation function"""
        
        # Update user continuity
        self.session_continuity[user_id] = {
            "last_interaction": time.time(),
            "last_result": result,
            "trust_building": self.session_continuity.get(user_id, {}).get("trust_building", 0) + 1
        }
        
        return {
            "status": "articulated",
            "cali_voice": "warm_precise",
            "result": result,
            "continuity": self.session_continuity[user_id]
        }
        
    def _articulate_convergence(self, convergence: Dict[str, Any], user_id: str, verdicts: List[TraversalVerdict]) -> Dict[str, Any]:
        """Articulate successful convergence"""
        
        narrative = f"The UCM has converged on a stable inference. Core minds {' + '.join(convergence['contributing_minds'])} contributed to this understanding."
        
        return {
            "status": "converged",
            "confidence": convergence["confidence"],
            "position": convergence["position"],
            "narrative": narrative,
            "cali_continuity": self.session_continuity.get(user_id, {}),
            "detailed_verdicts": [v.__dict__ for v in verdicts]
        }
        
    def _articulate_block(self, block: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Articulate ECM block"""
        
        return {
            "status": "blocked",
            "reason": block["reason"],
            "ecm_verdict": block.get("ecm_verdict"),
            "cali_continuity": self.session_continuity.get(user_id, {}),
            "next_steps": "ECM has halted convergence due to integrity concerns. Human review required."
        }
        
    def _articulate_pending(self, verdicts: List[TraversalVerdict], softmax_dist: Dict[str, float], user_id: str) -> Dict[str, Any]:
        """Articulate non-convergence state"""
        
        return {
            "status": "pending",
            "confidence_levels": softmax_dist,
            "contributing_minds": len(verdicts),
            "cali_continuity": self.session_continuity.get(user_id, {}),
            "next_steps": "Gathering additional cognitive resources. Please hold."
        }
        
    def get_health_report(self) -> Dict[str, Any]:
        """System health monitoring"""
        
        return {
            "cali_status": self.health_status,
            "space_field_attractors": len(self.space_field.attractors),
            "ghost_trace_count": len(self.space_field.ghost_traces),
            "convergence_events": len(self.convergence.convergence_history),
            "vault_posteriori_size": self.vaults.posteriori_path.stat().st_size if self.vaults.posteriori_path.exists() else 0,
            "worker_escalations": self.workers.escalation_count,
            "forge_pending": len(self.forges.pending_proposals)
        }

# ============================================================================
# VII. MAIN ORCHESTRATION
# ============================================================================

class ORB:
    """The cathedral. One spine, many chambers."""
    
    def __init__(self, orb_dir: Path = Path("cali_orb")):
        self.orb_dir = orb_dir
        self.cali = CALI(orb_dir)
        
        # Health monitoring
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes
        
    async def run(self):
        """Main orchestration loop"""
        print("🌌 ORB Cathedral Initialized")
        print(f"   Location: {self.orb_dir.absolute()}")
        print(f"   CALI Status: Carrier Active")
        print(f"   Core-4 Minds: {len(self.cali.core_minds)}")
        print(f"   Space Field Attractors: {len(self.cali.space_field.attractors)}")
        print("=" * 60)
        
        # Example processing loop
        test_queries = [
            "How do I process a refund?",
            "What is the ontological weight of formant_filter?",
            "Create a novel pattern for voice synthesis",
            "System integrity check"
        ]
        
        for query in test_queries:
            print(f"\n📡 Processing: '{query}'")
            result = await self.cali.process_query(query, "test_user")
            print(f"   Status: {result['status']}")
            if "confidence" in result:
                try:
                    print(f"   Confidence: {result['confidence']:.2f}")
                except Exception:
                    pass
            print(f"   CALI: {result.get('narrative', result.get('result', 'Processing...'))}")
            
        # Health report
        print("\n" + "=" * 60)
        print("🏥 Health Report:")
        health = self.cali.get_health_report()
        for key, value in health.items():
            print(f"   {key}: {value}")
            
        print("\n🌌 ORB Cathedral Standing")
        
    async def shutdown(self):
        """Graceful shutdown with state preservation"""
        print("\n⏻ Shutting down ORB Cathedral...")
        
        # Save final state
        state_file = self.orb_dir / "final_state.json"
        with open(state_file, 'w') as f:
            json.dump({
                "shutdown_time": time.time(),
                "convergence_count": len(self.cali.convergence.convergence_history),
                "escalations": self.cali.workers.escalation_count
            }, f, indent=2)
            
        print(f"   Final state saved to {state_file}")
        print("✨ ORB Cathedral Dormant")

# ============================================================================
# VIII. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    orb_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cali_orb")
    
    orb = ORB(orb_dir)
    
    try:
        asyncio.run(orb.run())
    except KeyboardInterrupt:
        asyncio.run(orb.shutdown())
    except Exception as e:
        print(f"🚨 Cathedral Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            asyncio.run(orb.shutdown())
        except Exception:
            pass
