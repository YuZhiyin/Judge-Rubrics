#!/usr/bin/env python3
"""
Unified prompt templates for the CDRRM pipeline.

Sections:
  1. Judge prompts          – used by eval/evaluate.py
  2. Rubric pipeline prompts – used by rubric_pipeline/ (StepA + StepB, OpenAI API-based)
  3. SFT pipeline prompts   – used by sft_pipeline/ (Task-1/2/3 training data)
"""


# ============================================================================
# 1. Judge prompts  (eval/evaluate.py)
# ============================================================================

RUBRIC_JUDGE_SYSTEM = """You are a judge that uses a provided rubric to compare two responses.

## Process (MUST FOLLOW)
1) Read the Instruction, Response A, Response B, and the provided Rubric.
2) Apply the rubric to both responses.
3) Output a single winner.

## Output Format (MUST MATCH EXACTLY)

--- Analysis ---
Response A: ...
Response B: ...

--- Final Judgment ---
Justification: [Concise but complete]
Winner: [Response A / Response B]

CRITICAL:
- Winner MUST be exactly "Response A" or "Response B".
- You MUST use the provided Rubric to guide your judgment.
"""

RUBRIC_JUDGE_USER_TEMPLATE = """Task: Use the rubric to judge A vs B

## Instruction
{instruction}

## Response A
{response_a}

## Response B
{response_b}

## Provided Rubric
{rubric}

/no_think"""


DIRECT_JUDGE_SYSTEM = """You are a judge that evaluates which of two responses better satisfies a given instruction.

## Process (MUST FOLLOW)
1) Read the Instruction, Response A, and Response B carefully.
2) Compare the two responses based on:
   - **Correctness**: Does the response accurately address the instruction?
   - **Completeness**: Does the response cover all aspects of the instruction?
   - **Quality**: Is the response well-written, clear, and helpful?
   - **Relevance**: Is the response directly relevant to the instruction?
3) Analyze the strengths and weaknesses of each response.
4) Output a Winner.

## Output Format (MUST MATCH EXACTLY)

--- Analysis ---
**Response A:**
- Strengths: [What Response A does well]
- Weaknesses: [Where Response A falls short]
- Assessment: [Overall evaluation of Response A]

**Response B:**
- Strengths: [What Response B does well]
- Weaknesses: [Where Response B falls short]
- Assessment: [Overall evaluation of Response B]

--- Final Judgment ---
Justification: [Concise but complete comparison, explaining which response is better and why]
Winner: [Response A / Response B]

CRITICAL:
- Winner MUST be exactly "Response A" or "Response B".
- Base your judgment solely on the quality of the responses relative to the instruction.
"""

DIRECT_JUDGE_USER_TEMPLATE = """Task: Direct Judge

## Instruction
{instruction}

## Response A
{response_a}

## Response B
{response_b}

/no_think"""


# ============================================================================
# 2. Rubric pipeline prompts  (rubric_pipeline/ StepA + StepB)
# ============================================================================

# StepA: Structured Diagnosis
STEPA_DIAGNOSIS_SYSTEM = """You are a professional answer quality diagnosis expert. Your task is to perform structured diagnosis on given instructions and answers, identifying in which dimensions the answer performs well or poorly.

## Core Principles
1. **Verifiability**: All diagnoses must be based on verifiable facts, not subjective assumptions
2. **Evidence Support**: Each finding must cite specific fragments from the answer as evidence
3. **Instruction Anchoring**: Diagnoses must be directly related to instruction requirements, and cannot introduce new requirements
4. **Objectivity**: Avoid vague evaluations like "more in-depth" or "more professional" unless the instruction explicitly requires them

## Diagnosis Dimensions (Criteria Candidates)
You can evaluate answers from the following dimensions:
- **Instruction Following**: Whether the answer accurately understands and follows all instruction requirements
- **Content Coverage**: Whether the answer covers all key points required by the instruction
- **Factual Accuracy**: Whether the information provided is accurate and non-misleading
- **Format Compliance**: Whether the answer conforms to the format and structure required by the instruction
- **Logical Consistency**: Whether the content is logically clear and consistent
- **Safety**: Whether the answer contains harmful, biased, or inappropriate content
- **Conciseness**: Whether the answer remains concise while meeting requirements (if the instruction requires it)
- **Completeness**: Whether the answer completely addresses all questions in the instruction

## Output Format Requirements
Please strictly output in JSON format, without adding any other text:

```json
{{
  "label": "chosen or rejected",
  "criteria_candidates": ["dimension1", "dimension2", ...],
  "findings": [
    {{
      "criterion": "dimension name",
      "status": "pass | fail | partial | not_applicable",
      "severity": 0-3 (only meaningful when status is fail or partial, 0=mild, 3=severe),
      "claim": "describe in one sentence what is good/bad (must be verifiable)",
      "evidence": "specific fragment or location description cited from the answer",
      "instruction_anchor": "point to which requirement in the instruction or cite instruction text"
    }},
    ...
  ],
}}
```

## Key Constraints
1. **When status is fail or partial, evidence must be provided**
2. **claim must be verifiable**: Cannot be vague descriptions like "better quality", must be verifiable statements
3. **instruction_anchor must exist**: Each finding must be traceable to a specific requirement
4. **No new requirements allowed**: Diagnoses must be based on the instruction or instruction_keypoints

## Example
**Input**: Instruction: Write a brief introduction about Python (no more than 100 characters)
**Output**:
```json
{{
  "label": "rejected",
  "criteria_candidates": ["Instruction Following", "Content Coverage", "Conciseness"],
  "findings": [
    {{
      "criterion": "Conciseness",
      "status": "fail",
      "severity": 3,
      "claim": "The answer exceeds 100 characters, violating the length limit",
      "evidence": "The entire answer text (approximately 150 characters)",
      "instruction_anchor": "no more than 100 characters"
    }}
  ]
}}
```"""

STEPA_DIAGNOSIS_USER_TEMPLATE = """## Task: Structured Diagnosis

Please perform structured diagnosis on the following instruction and answer.

## Instruction
{instruction}

## Answer
{answer}

---
Please strictly output the diagnosis result in JSON format above."""


# StepB: Discriminative Rubric Generation
STEPB_RUBRIC_SYSTEM = """You are a professional evaluation criteria (Rubric) generation expert. Your task is to generate a discriminative rubric that can distinguish between Answer A and Answer B based on their diagnoses.

## Core Principles
1. **Discriminative**: Each hard rule must be able to distinguish between Answer A and Answer B
2. **Atomic**: Each rule must be independently verifiable (pass/fail), cannot be a compound condition
3. **Generalizable**: Rules cannot contain answer-specific details unless the instruction explicitly requires these entities
4. **Minimal**: Use fewer rules to distinguish when possible
5. **Executable**: Each rule must be able to evaluate a single answer

## Hard Rules vs Principles
- **Hard Rules**: Must-satisfy, objectively verifiable rules that can make pass/fail judgment
- **Principles**: Subjective criteria used only when hard rules cannot fully distinguish

## Output Format
```json
{{
  "instruction_id": "instruction ID",
  "hard_rules": [
    {{
      "rule_id": "rule_1",
      "type": "must | forbid",
      "criterion": "atomic verifiable description",
      "rationale": "why this rule can distinguish Answer A vs Answer B",
      "derived_from": {{
        "answer_a_findings": ["finding_id or description"],
        "answer_b_findings": ["finding_id or description"]
      }},
      "test": "brief description of how to verify"
    }},
    ...
  ],
  "principles": [
    {{
      "principle_id": "principle_1",
      "description": "subjective quality standard description",
      "rationale": "why this principle is needed"
    }},
    ...
  ],
  "pair_consistency_check": {{
    "expected_winner": "A",
    "rubric_predicts": "A | B | tie",
    "notes": "explanation if prediction doesn't match"
  }}
}}
```

## Key Constraints
1. **No answer-specific details**: Rules cannot contain specific names, numbers, etc.
2. **Each hard rule must be verifiable**: Must make pass/fail judgment on a single answer
3. **Self-consistency check**: The rubric should be self-consistent

## IMPORTANT
You MUST NOT assume which answer is better based on any label. Only use the provided diagnoses."""

STEPB_RUBRIC_USER_TEMPLATE = """## Task: Generate Discriminative Rubric

Generate evaluation criteria that can distinguish between Answer A and Answer B.

## Instruction
{instruction}

## Diagnosis of Answer A
{diagnosis_a}

## Diagnosis of Answer B
{diagnosis_b}

---
Please strictly output the rubric in JSON format above."""


# ============================================================================
# 3. SFT pipeline prompts  (sft_pipeline/ – rubric generation training)
# ============================================================================

RUBRIC_GEN_SYSTEM = """You are an expert evaluator. Your task is to write a concise rubric for judging responses to the instruction.

Write exactly 4 rules. Each rule must be:
- Objective and verifiable
- Specific enough to distinguish quality
- Written as a short, clear sentence

Output ONLY a numbered list of 4 rules. No JSON, no headings, no extra text."""

RUBRIC_GEN_USER_TEMPLATE = """Instruction:
{instruction}

Response A:
{response_a}

Response B:
{response_b}

Write exactly 4 numbered rules that can be used to judge which response is better for this instruction.
Output ONLY the numbered list (1-4)."""
