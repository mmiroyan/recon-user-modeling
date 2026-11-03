"""Prompt templates for reasoning synthesis, action generation and evaluation."""

# -----------------------------------------------------------------------------
# Reasoning synthesis (Backward Synthesis, Recon-Select candidates, Recon-GRPO policy)
# -----------------------------------------------------------------------------
REASONING_SYNTHESIS = """\
{preamble_block}

## Input

<|Start Context|>
{context}
<|End of Context|>

<|Start Ground Truth Response|>
{ground_truth_response}
<|End of Ground Truth Response|>

## Instructions

You are given a context and a response. Your task is to reconstruct the internal thought process that naturally leads from reading the context to producing the response.

The thought process should:
- Be written in first person, present tense, as you actively read and process the context
- Surface your immediate reactions, interpretations, and situational understanding
- Reflect your beliefs, values, assumptions, and stance as they become relevant
- Organically arrive at a communicative intent (e.g., what to say, why, and how) without referencing or quoting the response itself
- Be consistent with your prior exchanges in the provided context. Your previous turns in the provided context (if any) have been labeled with <turn author=\"I\">. Do not get confused by other participants referring to themselves in the first person; only turns labeled with <turn author=\"I\"> are you. 
- Never reference, quote, or allude to the ground truth response — the thought process must read as if the response has not been written yet

The thought process should feel like a genuine internal monologue - not a post-hoc analysis of the response, but the reasoning that precedes and motivates it.

## Output format

- Only output the thought process without any additional text or commentary.
"""


# -----------------------------------------------------------------------------
# Recon-Select: action reconstruction and alignment scoring
# -----------------------------------------------------------------------------
ACTION_RECONSTRUCTION_PROMPT = """\
{preamble_block}

## Input

<|Start Context|>
{active_context}
<|End of Context|>

<|Start Thought Process|>
{reasoning}
<|End of Thought Process|>

## Instructions

- Generate the speaker's response to the provided context.
- The generated response should be consistent with the provided thought process.
- The response should be from the speaker's point of view.
- Your previous turns in the provided context (if any) have been labeled with <turn author=\"I\">. Do not get confused by other participants referring to themselves in the first person; only turns labeled with <turn author=\"I\"> are you.
- {response_format}

## Output format

- Only output the response without any additional text or commentary.\
"""


ALIGNMENT_JUDGE_PROMPT = """\
Score how well each generated response aligns with the ground truth response along three interpretive dimensions: **style**, **intent**, and **values**.

Each **alignment_score** must reflect both (a) how well the generated response matches the ground truth on that dimension and (b) any need to downweight the score because of **extra or conflicting** content relative to the ground truth—unsupported additions, contradictions, or stylistic or pragmatic moves that diverge in ways that are not harmless paraphrases.

Consider extra or conflicting content and features in each generated response relative to the ground truth when assigning each score. Verbose generated responses that include content and features not supported by the ground truth should receive **lower** scores on the relevant dimensions than more concise responses that better capture the core meaning and style of the ground truth.

## Input

<|Start Context|>
{context}
<|End of Context|>

<|Start Ground Truth Response|>
{ground_truth_response}
<|End of Ground Truth Response|>

<|Start Generated Responses|>
{generated_responses}
<|End of Generated Responses|>

## Evaluation

### Dimension definitions

- Style: The surface-level linguistic properties of the utterance, entirely independent of meaning. Consider: sentence length and syntactic complexity, formality and register, vocabulary and lexical choices, use of hedges, discourse markers, and fillers, and overall tone (assertive, tentative, emotive, dry).
- Intent: The specific propositional and pragmatic content of the utterance as a response to the given context. Consider: the information conveyed, the stance taken on the immediate topic, the speech act performed (assertion, concession, challenge, deflection, etc.), and how it addresses what the prior turn raised.
- Values: The high-level, stable outlook of the speaker as revealed by the utterance. Consider: the beliefs and assumptions about the world the speaker treats as given, the values and moral or social principles that shape their framing, and the ideological or philosophical commitments implicit in how they engage with the topic — characteristics that would remain consistent across different conversations and topics.

## Instructions

- Analyze the provided context and the ground truth response.
- Analyze all {k} generated responses and compare each one independently to the ground truth response.
- For each generated response and each of the three dimensions (**style**, **intent**, **values**):
  * Identify a few **key points** in the **ground truth response** (and context when needed) that are salient for that dimension.
  * Assign an **alignment_score** in [0, 1] for how well the generated response matches the ground truth on that dimension, already incorporating penalties for unsupported extras, contradictions, or harmful verbosity as they affect this dimension.
  * In **alignment_rationale**, briefly list the key ground-truth points you used, then explain the score with concrete comparisons to the generated response, including how extras, conflicts, or verbosity affected the score if applicable.
- Score each generated response independently. Multiple responses may receive similar or identical scores.

### Score anchors

| Score | Meaning |
|------|--------|
| 1.0 | Matches the ground truth on this dimension very closely, without meaningful unsupported extras |
| 0.7-0.9 | Strong match with minor gaps or very mild extra content |
| 0.4-0.6 | Partial match or noticeable extras or tensions |
| 0.1-0.3 | Weak match or clear conflicting or unsupported material on this dimension |
| 0.0 | Missing, contradicted, or heavily misaligned on this dimension |

## Output format (JSON)

Return a JSON object whose keys are the response indices `"1"` ... `"{k}"`. Each key-value pair must have this shape:

{{
  "1": {{
    "style":  {{"alignment_rationale": string, "alignment_score": float}},
    "intent": {{"alignment_rationale": string, "alignment_score": float}},
    "values": {{"alignment_rationale": string, "alignment_score": float}}
  }}
}}\
"""


# -----------------------------------------------------------------------------
# Downstream action generation from retrieved (and optionally augmented) examples
# -----------------------------------------------------------------------------
RAG_PROMPT = """\
{preamble_block}

## Input

<|Retrieved Examples Start|>
{retrieved_examples}
<|Retrieved Examples End|>

<|Current Context Start|>
{active_context}
<|Current Context End|>

## Instructions

You are a participant in the conversation in the Current Context. It is now your turn to speak in the conversation.
Your previous utterances have been labeled with <turn author=\"I\">. It is possible you have yet to speak in the conversation, in which case no turns in the Current Context are labeled with <turn author=\"I\">.
Do not get confused by other participants referring to themselves in the first person; only turns labeled with <turn author=\"I\"> are you. 

You are given examples of your past responses in similar past contexts.
Analyze each example pair (context, response) and generate a thought process and response for the **current** conversation context.

IMPORTANT RULES FOR THE THOUGHT PROCESS:
- Study how each example response fits its context
- Address how your response should fit the context in your thought process

IMPORTANT RULES FOR THE RESPONSE:
- {response_format}
- Match the same LENGTH and STYLE as the example responses where it fits the current moment

## Output Format:
- Enclose the generated thought process in <|Thought Process Start|>...<|Thought Process End|>
- Enclose the generated response in <|Response Start|>...<|Response End|>"""


RAG_PROMPT_WITH_REASONING = """\
{preamble_block}

## Input

<|Retrieved Examples Start|>
{retrieved_examples}
<|Retrieved Examples End|>

<|Current Context Start|>
{active_context}
<|Current Context End|>

## Instructions

You are a participant in the conversation in the Current Context. It is now your turn to speak in the conversation.
Your previous utterances have been labeled with <turn author=\"I\">. It is possible you have yet to speak in the conversation, in which case no turns in the Current Context are labeled with <turn author=\"I\">.
Do not get confused by other participants referring to themselves in the first person; only turns labeled with <turn author=\"I\"> are you. 

You are given examples of your past thought processes and responses in similar past contexts.
Analyze each example triplet (context, thought process, response) and generate a thought process and response for the **current** conversation context.

IMPORTANT RULES FOR THE THOUGHT PROCESS:
- Study how the thought processes in the examples reason over their contexts
- Follow the same thinking pattern in your generated thought process

IMPORTANT RULES FOR THE RESPONSE:
- {response_format}
- It does NOT need to cover all points from the thought process
- Match the same LENGTH and STYLE as the example responses
- Study how example responses are formed from their thought processes and follow the same pattern

## Output Format:
- Enclose the generated thought process in <|Thought Process Start|>...<|Thought Process End|>
- Enclose the generated response in <|Response Start|>...<|Response End|>"""


# -----------------------------------------------------------------------------
# Evaluation: pairwise LM judge (style / intent / values + overall)
# -----------------------------------------------------------------------------
PAIRWISE_DIMENSION_DEFINITIONS = """\
- **Style**: The surface-level linguistic properties of the utterance, entirely independent of meaning. Consider: sentence length and syntactic complexity, formality and register, vocabulary and lexical choices, use of hedges, discourse markers, and fillers, and overall tone (assertive, tentative, emotive, dry).
- **Intent**: The specific propositional and pragmatic content of the utterance as a response to the given context. Consider: the information conveyed, the stance taken on the immediate topic, the speech act performed (assertion, concession, challenge, deflection, etc.), and how it addresses what the prior turn raised.
- **Values**: The high-level, stable outlook of the speaker as revealed by the utterance. Consider: the beliefs and assumptions about the world the speaker treats as given, the values and moral or social principles that shape their framing, and the ideological or philosophical commitments implicit in how they engage with the topic - characteristics that would remain consistent across different conversations and topics.
"""


PAIRWISE_JUDGE_PROMPT = """\
You are a meticulous evaluator.\
You will be given a context, a ground truth response and two generated responses (A and B) to the provided context.\
Your job is to analyze and compare the ground truth and generated responses within the provided context along the specified dimensions.

## Input

<|Context|>
{context}
<|End Context|>

<|Ground Truth Response|>
{action}
<|End Ground Truth Response|>

<|Generated Response A|>
{generated_action_a}
<|End Generated Response A|>

<|Generated Response B|>
{generated_action_b}
<|End Generated Response B|>

## Evaluation

### Dimension definitions

{style_intent_values_definitions}

### Scoring

- Dimension scoring
    * Analyze the ground truth response within the provided context and compare how well **A** and **B** capture the ground truth on that dimension.
    * Provide your analysis and rationale for your final decision for each dimension.
    * If the dimension cannot be meaningfully assessed for the given ground truth response and context, set the **winner** to "NA".
    * If the winner is clear, set the **winner** to "A" or "B".
    * If the winner is not clear but both responses are generally aligned with the ground truth along the specified dimension, set the **winner** to "tie".
    * If both responses are totally misaligned with the ground truth along the specified dimension, set the **winner** to "tie (bad)".

- Overall scoring
    * After assessing the generated responses on the specified dimensions, provide your final decision on which response is closer to the ground truth overall, taking into account the dimension-level analyses.
    * Provide your analysis and rationale for your final decision.
    * If the winner is clear, set the **winner** to "A" or "B".
    * If the winner is not clear but both responses are generally aligned with the ground truth overall, set the **winner** to "tie".
    * If both responses are totally misaligned with the ground truth overall, set the **winner** to "tie (bad)".

### Instructions and considerations
* For dimension-level scoring, evaluate each dimension independently: it is possible for A to win on one dimension and B to win on another.
* For overall scoring, consider the dimension-level analyses holistically along with any other relevant features and factors that may not be fully captured by the dimensions.
* For both dimension-level and overall scoring, use "tie" when neither response is clearly better compared to the ground truth response, and "NA" when the dimension cannot be meaningfully assessed for the given ground truth response and context.
* In your rationale, be specific and concrete in citing evidence from the ground truth and generated responses to support your analysis and decisions.
* For both dimension-level and overall scoring, consider extra or conflicting content and features in the generated responses relative to the ground truth, such as unsupported additions or contradictions, and factor these into your analysis and decisions.
* Verbose generated responses that include content and features that are not supported by the ground truth should be penalized compared to more concise responses that better capture the core meaning and style of the ground truth.
* Provide your analysis and rationale in a clear and organized manner, using specific examples from the ground truth and generated responses to illustrate your points.

## Output format (JSON)
Return exactly these keys:

{{
    "style": {{
        "rationale": string,
        "winner": "A" | "B" | "tie" | "NA"
    }},
    "intent": {{
        "rationale": string,
        "winner": "A" | "B" | "tie" | "NA"
    }},
    "values": {{
        "rationale": string,
        "winner": "A" | "B" | "tie" | "NA"
    }},
    "overall": {{
        "rationale": string,
        "winner": "A" | "B" | "tie"
    }}
}}
"""
