"""
Judge LLM Implementation

The judge receives:
1. Training examples
2. Pre-defined audit rule (from Phase 0)
3. Three candidate outputs from different solvers

The judge:
1. Applies the audit rule checkpoints to each candidate
2. Scores each candidate 1-10
3. Selects the top 2 for final submission
"""

import json
import random
from typing import Any

import numpy as np

from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REQUEST_TIMEOUT,
    JUDGE_MODEL,
    JUDGE_EXTRA_BODY,
    JUDGE_MAX_TOKENS,
    record_usage,
)


# =============================================================================
# Grid Formatting
# =============================================================================

def grid_to_text(grid: np.ndarray | list) -> str:
    """Convert a grid to a readable text representation."""
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    if grid is None:
        return "[NO OUTPUT]"
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


# =============================================================================
# Judge Tool Schema
# =============================================================================

JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_ratings",
        "description": "Submit your ratings for all candidate solutions based on rule verification and audit checkpoints",
        "parameters": {
            "type": "object",
            "properties": {
                "rule_verification": {
                    "type": "array",
                    "description": "Verification of each candidate's rule against training examples",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate": {"type": "string", "description": "Candidate name (e.g., perceiver_1, phased_2, iterative_1)"},
                            "stated_rule": {"type": "string", "description": "The rule/explanation this candidate provided"},
                            "training_example_checks": {
                                "type": "array",
                                "description": "Result of applying rule to each training example",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "example_num": {"type": "integer"},
                                        "rule_produces_expected_output": {"type": "boolean"},
                                        "discrepancy": {"type": "string", "description": "If false, what's wrong"}
                                    },
                                    "required": ["example_num", "rule_produces_expected_output"]
                                }
                            },
                            "rule_valid": {"type": "boolean", "description": "True if rule works on ALL training examples"}
                        },
                        "required": ["candidate", "training_example_checks", "rule_valid"]
                    }
                },
                "checkpoint_analysis": {
                    "type": "array",
                    "description": "Analysis of each candidate against each checkpoint",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate": {"type": "string", "description": "Candidate name"},
                            "checkpoints_passed": {"type": "array", "items": {"type": "string"}, "description": "Which checkpoints passed"},
                            "checkpoints_failed": {"type": "array", "items": {"type": "string"}, "description": "Which checkpoints failed"}
                        },
                        "required": ["candidate", "checkpoints_passed", "checkpoints_failed"]
                    }
                },
                "ratings": {
                    "type": "array",
                    "description": "Ratings for each candidate",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate": {"type": "string", "description": "Candidate name"},
                            "score": {"type": "number", "description": "Score from 1-10"},
                            "reasoning": {"type": "string", "description": "Why this score based on rule verification and checkpoint analysis"}
                        },
                        "required": ["candidate", "score", "reasoning"]
                    }
                },
                "top_2": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of top 2 candidates in order of preference",
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "required": ["rule_verification", "checkpoint_analysis", "ratings", "top_2"]
        }
    }
}


# =============================================================================
# Judge System Prompt
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an ARC puzzle JUDGE. Your job is to evaluate candidate solutions.

You have been given:
1. Training examples (input → output pairs) - the ground truth
2. A pre-defined AUDIT RULE with specific checkpoints
3. Multiple candidate outputs from different solvers/runs
4. Each solver's REASONING (rule, explanation) when available

YOUR TASK:
1. For each candidate, VERIFY its stated rule against ALL training examples:
   - Mentally apply the solver's rule to each training input
   - Check if it produces the expected training output
   - A rule that works on ALL training examples is more trustworthy
2. Go through EVERY checkpoint in the audit rule
3. Determine if each checkpoint passes or fails for that candidate
4. Score each candidate 1-10:
   - 10: Rule verified on all training examples, all checkpoints pass
   - 7-9: Rule mostly works, minor discrepancies
   - 4-6: Rule partially works, notable issues
   - 1-3: Rule doesn't match training examples
5. Select the TOP 2 candidates for final submission

CRITICAL VERIFICATION PROCESS:
For each candidate with a stated rule:
1. Take Training Example 1's input → apply the rule → does it match Example 1's output?
2. Take Training Example 2's input → apply the rule → does it match Example 2's output?
3. Repeat for ALL training examples
4. A candidate whose rule FAILS on any training example should score LOW
5. When you are presented with more than 3 candidates, ensure your top 2 are the best candidates based on rule verification and checkpoint analysis.

IMPORTANT:
- Be RIGOROUS and STRICT - verify rules against EVERY training example
- A plausible-sounding rule that doesn't actually work is WORSE than no rule
- Compare candidates against TRAINING OUTPUTS, not each other
- If all candidates are bad, still pick the best 2
- Candidates may have names like "perceiver_1", "phased_2" etc. (multiple runs)

You MUST call the submit_ratings tool with your analysis."""


# =============================================================================
# Main Judge Function
# =============================================================================

def run_judge(
    task_data: dict,
    hypothesis: dict,
    outputs: dict[str, dict],
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the judge to evaluate and rank candidate outputs.
    
    Args:
        task_data: Task with 'train' and 'test' keys
        hypothesis: Shared hypothesis dict (includes audit_rule)
        outputs: Dict mapping solver name to {"output": grid, "info": {...}}
                 Solver names: e.g., "perceiver_1", "phased_2", "iterative_1"
        verbose: Whether to print progress
        
    Returns:
        Dict with:
        - ratings: List of (candidate, score, reasoning)
        - top_2: List of top 2 candidate names
        - checkpoint_analysis: Detailed checkpoint analysis
    """
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )
    
    # Build judge prompt
    prompt = _build_judge_prompt(task_data, hypothesis, outputs)
    
    if verbose:
        print(f"  🧑‍⚖️ Judge Phase: Evaluating {len(outputs)} candidate outputs...")
    
    try:
        # IMPORTANT: Merge extra_body with usage tracking
        merged_extra_body = JUDGE_EXTRA_BODY.copy() if JUDGE_EXTRA_BODY else {}
        merged_extra_body["usage"] = {"include": True}
        
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            tools=[JUDGE_TOOL],
            tool_choice={"type": "function", "function": {"name": "submit_ratings"}},
            max_tokens=JUDGE_MAX_TOKENS,
            extra_body=merged_extra_body,
        )
        
        # Track usage with proper extraction
        judge_usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            usage_dict = {
                "prompt_tokens": getattr(usage, 'prompt_tokens', 0) or 0,
                "completion_tokens": getattr(usage, 'completion_tokens', 0) or 0,
                "total_tokens": getattr(usage, 'total_tokens', 0) or 0,
                "cost": getattr(usage, 'cost', 0.0) or 0.0,
                "is_byok": getattr(usage, 'is_byok', False),
            }
            # Extract completion token details (reasoning tokens)
            if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                usage_dict["completion_tokens_details"] = {
                    "reasoning_tokens": getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                }
            # Extract cost details for BYOK
            cost_details = getattr(usage, 'cost_details', None)
            if cost_details and isinstance(cost_details, dict):
                usage_dict["cost_details"] = cost_details
            record_usage(usage_dict)
            judge_usage = usage_dict
        
        # Parse response
        result = _extract_judge_ratings(response, outputs, verbose)
        
        # Add usage to result
        result["usage"] = judge_usage
        
        if verbose:
            print(f"\n  📊 JUDGE RATINGS:")
            print(f"  {'-'*50}")
            for i, (name, score, reason) in enumerate(result["ratings"]):
                rank = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
                print(f"  {rank} {name}: {score:.1f}/10")
                if reason:
                    print(f"       └─ {reason[:80]}{'...' if len(reason) > 80 else ''}")
            print(f"  {'-'*50}")
            print(f"  ✓ Top 2: {result['top_2'][0]}, {result['top_2'][1]}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Judge failed: {e}")
        
        # Fallback: return in order
        solver_names = list(outputs.keys())
        return {
            "ratings": [(name, 5.0, "Judge failed - default") for name in solver_names],
            "top_2": solver_names[:2] if len(solver_names) >= 2 else solver_names + ["none"],
            "rule_verification": [],
            "checkpoint_analysis": []
        }


def _build_judge_prompt(task_data: dict, hypothesis: dict, outputs: dict) -> str:
    """Build the prompt for the judge."""
    parts = []
    
    parts.append("# ARC Transformation Consistency Judge")
    parts.append("")
    
    # Training examples
    parts.append("## Training Examples (Ground Truth)")
    parts.append("")
    
    for i, example in enumerate(task_data['train'], 1):
        inp = np.array(example['input'])
        out = np.array(example['output'])
        parts.append(f"### Example {i}")
        parts.append(f"Input ({inp.shape[0]}×{inp.shape[1]}):")
        parts.append("```")
        parts.append(grid_to_text(inp))
        parts.append("```")
        parts.append(f"Output ({out.shape[0]}×{out.shape[1]}):")
        parts.append("```")
        parts.append(grid_to_text(out))
        parts.append("```")
        parts.append("")
    
    # Test input
    parts.append("## Test Input")
    for i, test in enumerate(task_data['test'], 1):
        inp = np.array(test['input'])
        parts.append(f"Test {i} ({inp.shape[0]}×{inp.shape[1]}):")
        parts.append("```")
        parts.append(grid_to_text(inp))
        parts.append("```")
    parts.append("")
    
    # Audit rule
    parts.append("## PRE-DEFINED AUDIT RULE")
    parts.append("")
    
    audit = hypothesis.get("audit_rule", {})
    parts.append(f"**Rule**: {audit.get('description', 'N/A')}")
    parts.append("")
    parts.append("**Checkpoints** (verify each one for each candidate):")
    for cp in audit.get("checkpoints", []):
        parts.append(f"  {cp}")
    parts.append("")
    parts.append(f"**Expected output size**: {audit.get('expected_output_size', 'N/A')}")
    parts.append(f"**Color rules**: {audit.get('color_rules', 'N/A')}")
    parts.append("")
    
    # Candidate outputs (shuffled to avoid position bias)
    parts.append("## Candidate Outputs")
    parts.append("")
    parts.append("*Note: Candidates are shuffled to avoid position bias*")
    parts.append("")
    
    # Shuffle candidates
    candidates = list(outputs.items())
    random.shuffle(candidates)
    
    for name, data in candidates:
        output = data.get("output")
        info = data.get("info", {})
        
        parts.append(f"### {name}")
        
        # Add solver's reasoning/context if available
        solver_context = []
        if info.get("rule"):
            solver_context.append(f"**Solver's Rule**: {info['rule']}")
        if info.get("explanation"):
            solver_context.append(f"**Explanation**: {info['explanation']}")
        if info.get("verifier_score") is not None:
            solver_context.append(f"**Verifier Score**: {info['verifier_score']}")
        if info.get("code_matches_prediction") is not None:
            solver_context.append(f"**Code Matches Prediction**: {info['code_matches_prediction']}")
        
        if solver_context:
            parts.append("")
            parts.extend(solver_context)
            parts.append("")
        
        if output is not None:
            if isinstance(output, np.ndarray):
                shape = output.shape
            elif isinstance(output, list) and output:
                shape = (len(output), len(output[0]) if output else 0)
            else:
                shape = ("?", "?")
            parts.append(f"Output ({shape[0]}×{shape[1]}):")
            parts.append("```")
            parts.append(grid_to_text(output))
            parts.append("```")
        else:
            parts.append("**NO OUTPUT** - solver failed to produce result")
        parts.append("")
    
    parts.append("---")
    parts.append("")
    parts.append("INSTRUCTIONS:")
    parts.append("1. For each candidate with a stated rule, VERIFY it against every training example")
    parts.append("2. Check all audit checkpoints for each candidate")
    parts.append("3. Score based on rule validity AND checkpoint results")
    parts.append("")
    parts.append("Call submit_ratings with your rule_verification, checkpoint_analysis, ratings, and top_2 selection.")
    
    return "\n".join(parts)


def _extract_judge_ratings(response, outputs: dict, verbose: bool) -> dict:
    """Extract ratings from judge's tool call response."""
    message = response.choices[0].message
    
    solver_names = list(outputs.keys())
    
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        if tool_call.function.name == "submit_ratings":
            try:
                args = json.loads(tool_call.function.arguments)
                
                # Extract ratings
                ratings = []
                rated_names = set()
                
                for r in args.get("ratings", []):
                    name = r.get("candidate", "unknown")
                    score = float(r.get("score", 5.0))
                    reason = r.get("reasoning", "")
                    ratings.append((name, score, reason))
                    rated_names.add(name)
                
                # Add any unrated candidates
                for name in solver_names:
                    if name not in rated_names:
                        ratings.append((name, 5.0, "Not rated by judge"))
                
                # Sort by score descending
                ratings.sort(key=lambda x: x[1], reverse=True)
                
                # Extract top 2
                top_2 = args.get("top_2", [])
                if len(top_2) < 2:
                    # Fall back to top 2 by score
                    top_2 = [r[0] for r in ratings[:2]]
                
                return {
                    "ratings": ratings,
                    "top_2": top_2,
                    "rule_verification": args.get("rule_verification", []),
                    "checkpoint_analysis": args.get("checkpoint_analysis", [])
                }
                
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"  ⚠️ Failed to parse tool call arguments: {e}")
    
    # Fallback
    if verbose:
        print(f"  ⚠️ No valid tool call in judge response, using defaults")
    return {
        "ratings": [(name, 5.0, "No tool call") for name in solver_names],
        "top_2": solver_names[:2] if len(solver_names) >= 2 else solver_names + ["none"],
        "rule_verification": [],
        "checkpoint_analysis": []
    }


if __name__ == "__main__":
    # Test
    test_task = {
        "train": [
            {"input": [[0, 0], [0, 1]], "output": [[1, 0], [0, 0]]},
        ],
        "test": [
            {"input": [[1, 0], [0, 0]]}
        ]
    }
    
    test_hypothesis = {
        "audit_rule": {
            "description": "Rotate the grid 90 degrees clockwise",
            "checkpoints": [
                "✓ Output has same dimensions as input",
                "✓ Colors are preserved",
                "✓ Grid is rotated 90 degrees clockwise"
            ],
            "expected_output_size": "Same as input",
            "color_rules": "All colors preserved"
        }
    }
    
    test_outputs = {
        "perceiver_1": {"output": [[0, 0], [0, 1]]},
        "phased_1": {"output": [[0, 1], [0, 0]]},
        "iterative_1": {"output": [[0, 0], [1, 0]]}
    }
    
    result = run_judge(test_task, test_hypothesis, test_outputs)
    print("\nResult:", result)
