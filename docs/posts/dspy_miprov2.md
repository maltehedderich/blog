---
title: "Programmatic Prompt Optimization: Building a Spam Filter with DSPy and MIPROv2"
date: 2026-02-09
categories:
  - DSPy
  - LLM
  - Optimization
tags:
  - Advanced
  - Tutorial
draft: false
---

# Programmatic Prompt Optimization: Building a Spam Filter with DSPy and MIPROv2

Most prompt engineering today is manual and intuition-driven. We tweak wording, rearrange instructions, and evaluate outputs by gut feel. But as LLM applications move from prototypes to production, this approach doesn't scale. What if we could replace intuition with data, optimizing prompts programmatically against measurable metrics just like we tune any other part of a software system?

<!-- more -->

This is exactly what [DSPy](https://github.com/stanfordnlp/dspy) enables. Instead of manually writing prompts, you define the _signature_ of your task and a _metric_ for success, and let an optimizer find the best instructions and few-shot examples for you.

In this post, we’ll build a spam filter that analyzes raw headers (SPF, DKIM) alongside message text, using the `MIPROv2` optimizer to tune the performance.

## Tech Stack

- **[DSPy](https://github.com/stanfordnlp/dspy)**: The framework for programmatically optimizing prompts based on defined signatures and metrics.
- **[Phoenix](https://github.com/Arize-ai/phoenix)** (by Arize): For observability, tracing, and managing the datasets used for training and evaluation.
- **[OpenRouter](https://openrouter.ai/)**: To quickly try out different models. In this setup, we use `grok-4.1-fast` for the task execution and `claude-opus-4.6` as the "Teacher" to optimize the prompts.

## Defining the Signature

In DSPy, we start by defining _what_ the model should do, decoupled from _how_ it should be prompted. This is called a `Signature`. Each signature specifies the input fields, output fields, and their types.

```python
import dspy
from typing import Literal

SpamLevel = Literal[
    "not_spam",
    "unlikely_spam",
    "suspicious",
    "likely_spam",
    "very_likely_spam",
    "definitely_spam",
]

class SpamClassification(dspy.Signature):
    text: str = dspy.InputField(desc="The plaintext body of the email")
    subject: str = dspy.InputField(desc="The subject line")
    return_path: str | None = dspy.InputField(desc="The Return-Path header address")
    from_address: str | None = dspy.InputField(desc="The visible From header address")
    received_spf: str | None = dspy.InputField(desc="SPF verification result")
    reply_address: str | None = dspy.InputField(desc="The Reply-To address if different from the sender")
    authentication_results: str | None = dspy.InputField(desc="DKIM, SPF, DMARC results")

    reasons: list[str] = dspy.OutputField(desc="Specific red flags justifying the classification")
    classification: SpamLevel = dspy.OutputField(desc="Final determination of the spam risk level")


class SpamClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(SpamClassification)

    def forward(self, **kwargs):
        return self.classify(**kwargs)
```

Including fields like `received_spf` and `authentication_results` gives the model the same email authentication metadata that real mail servers rely on to detect spoofing (where an attacker forges the sender address to make a message appear to come from a trusted source). These headers are outside the scope of this post, but Microsoft has a [solid overview](https://learn.microsoft.com/en-us/defender-office-365/email-authentication-about/) if you're curious.

## Encoding Business Logic into Metrics

This is the most critical part of the DSPy workflow. Instead of eyeballing responses, we define a mathematical function for success.

In spam detection, errors have asymmetric costs. A **False Positive** (marking a legitimate email as spam) is a critical failure; the user might miss an important communication. A **False Negative** (letting a spam email through) is merely an annoyance.

Our metric explicitly encodes this trade-off:

```python

LEVEL_MAP = {
    "not_spam": 0,
    "unlikely_spam": 1,
    "suspicious": 2,
    "likely_spam": 3,
    "very_likely_spam": 4,
    "definitely_spam": 5,
}

def spam_metric(example: dspy.Example, prediction: dspy.Example, trace = None) -> float:
    """Calculate a weighted score based on classification correctness and prediction confidence.

    Args:
        example: The ground-truth example containing the expected classification.
        prediction: The model's predicted output with classification and confidence.
        trace: Optional execution trace used during optimization.

    Returns:
        A float score between 0.0 and 1.0, combining a decision score (70%)
        and a calibration bonus (30%).
    """

    # Only "very_likely_spam" and "definitely_spam" trigger the spam decision.
    # We set the bar high to minimize false positives on the binary junk/keep decision.
    pred_is_spam: bool = LEVEL_MAP.get(prediction.classification, 3) > 3
    is_spam: bool = example.classification == "definitely_spam"

    if pred_is_spam == is_spam:
        decision_score: float = 1.0  # Correct classification
    elif pred_is_spam and not is_spam:
        decision_score = 0.0  # False Positive (CRITICAL FAILURE)
    else:
        decision_score = 0.3  # False Negative (Tolerable)

    # Reward predictions that land close to the ground-truth level
    pred_level = LEVEL_MAP.get(prediction.classification, 3)
    true_level = LEVEL_MAP.get(example.classification, 0)
    calibration_bonus = 1.0 - abs(pred_level - true_level) / 5.0

    return 0.7 * decision_score + 0.3 * calibration_bonus
```

The trace parameter is useful when you want to validate intermediate steps during optimization to ensure your multi-hop reasoning program doesn't generate redundant or overly long queries.

## Dataset

I built 200 examples (balanced spam/non-spam) from my own inbox, which is why I won't share it publicly. You can create your own by annotating emails or using public corpora like the [Enron dataset](https://www.cs.cmu.edu/~enron/).

Here's what one training example looks like:

**Input**

```json
{
  "text": "I am Mr. Chris Brownridge from Rolls Royce Automobile Company UK. This is to officially announce to you that your E-mail Address has been selected among the winners of the latest 2024 ROLLS ROYCE CULLINAN MANSORY LUXURY SUV and A Check of (Ј15,000,000). This was carried out through the Rolls Royce online lottery promoter which the lucky winners are being selected through the Online randomly selection of Emails and you are among the 5th luckiest people to acquire this winnings, so for a properly verification and a successful delivery of this winnings, you are advised to provide your full information such as; Your Full Name: Your Full Address: Your Phone Number: Your Driver License: Contact us here on our email: infor@rrcompany1.wecom.work I'm respectively waiting for your response as soon as you receive my email Asap. Yours sincerely Mr. Chris Brownridge Rolls Royce Motor Cars Limited Office",
  "subject": "rom Rolls Royce Automobile Company UK.",
  "return_path": "<info@earthlink.net>",
  "from_address": "\"INFO ROLLS\"<info@earthlink.net>",
  "received_spf": "fail (spfCheck: domain of earthlink.net does not designate
 45.76.207.151 as permitted sender) client-ip=45.76.207.151;
 envelope-from=info@earthlink.net; helo=sms1004.com;",
  "reply_address": "<infor@rrcompany1.wecom.work>",
  "authentication_results": "amazonses.com;
 spf=fail (spfCheck: domain of earthlink.net does not designate 45.76.207.151
 as permitted sender) client-ip=45.76.207.151;
 envelope-from=info@earthlink.net; helo=sms1004.com;
 dmarc=fail header.from=earthlink.net;"
}
```

**Output**

```json
{
  "is_spam": true
}
```

Notice the label is binary, but our signature predicts six levels. Ideally, your dataset should match the target, but in reality, we rarely work with perfect conditions. The metric bridges the gap by mapping six-level predictions to binary decisions for scoring, while still rewarding calibration. MIPROv2 uses that signal to discover richer classification strategies than the labels alone could teach.

Phoenix makes it easy to manage datasets and track experiments. The dataset can be accessed via the Phoenix client:

```python
from phoenix.client import Client

phoenix_base_url = "<YOUR_PHOENIX_BASE_URL>"
phoenix_api_key = "<YOUR_PHOENIX_API_KEY>"
phoenix_client = Client(base_url=phoenix_base_url, api_key=phoenix_api_key)

dataset = phoenix_client.datasets.get_dataset(dataset="spam-classification", timeout=300)
```

Then we transform each record into a `dspy.Example` whose fields match the signature we defined earlier:

```python
trainset = []

for example in dataset.examples:
    inputs = {
            "text": example["input"]["text"],
            "subject": example["input"]["subject"],
            "return_path": example["input"].get("return_path"),
            "from_address": example["input"].get("from_address"),
            "received_spf": example["input"].get("received_spf"),
            "reply_address": example["input"].get("reply_address"),
            "authentication_results": example["input"].get("authentication_results"),
        }
    outputs = {
            "reasons": [],
            "classification": "definitely_spam" if example["output"]["is_spam"] else "not_spam",
        }
    trainset.append(dspy.Example(**inputs, **outputs).with_inputs(*inputs.keys()))
```

## Optimization with MIPROv2

With our data, module, and metric in place, we hand it to the **MIPROv2** (**M**ultiprompt **I**nstruction **PR**oposal **O**ptimizer Version 2).

MIPROv2 jointly optimizes _instructions_ and _few-shot examples_ for every **predictor** (a module that calls an LLM) in your program. Under the hood it runs three stages:

1.  **Bootstrap few-shot candidates**: The teacher model (`claude-opus-4.6`) runs the task on training examples, collecting input-output traces that serve as candidate demonstrations.
2.  **Propose instructions**: Using the bootstrapped traces and the training data, the teacher generates multiple instruction candidates grounded in the actual failure modes and dynamics of the task.
3.  **Bayesian Optimization**: Over a series of trials, MIPROv2 searches for the best _combination_ of instructions and demonstrations (few-shot examples) across all predictors. Each trial evaluates a candidate prompt set against `spam_metric` on a minibatch of the training data, and the best-averaging configuration is periodically validated on the full set.

```python
teacher_lm = dspy.LM("openrouter/x-ai/grok-4.1-fast", api_base=llm_base_url, api_key=openrouter_api_key)
student_lm = dspy.LM("openrouter/anthropic/claude-opus-4.6", api_base=llm_base_url, api_key=openrouter_api_key)  # Stronger model for optimization

optimizer = dspy.MIPROv2(
    metric=spam_metric,
    auto="medium",              # balances search depth vs. cost (light | medium | heavy)
    prompt_model=teacher_lm,    # Claude Opus 4.6 — proposes instructions & bootstraps demos
    task_model=student_lm,      # Grok 4.1 Fast  — executes the optimized prompts at inference
    num_threads=10              # How many parallel trials to run during optimization
)

optimized_app = optimizer.compile(SpamClassifier(), trainset=trainset)
```

The `auto` parameter controls how many instruction candidates, few-shot examples, and Bayesian trials MIPROv2 explores. `"medium"` is a practical sweet spot: thorough enough to find meaningful gains, without burning through your API budget on exhaustive search. For context, a single optimization run involved **~1,000 API calls** and processed **~6 million tokens**, costing approximately **$5 USD** via OpenRouter.

What comes back isn't just a "better prompt"; it's a fully parameterized configuration (instructions + demonstrations) for the student model that has been empirically optimized against your metric and your data.

```json
{
  "classify": {
    "demos": [
      {
        "augmented": true,
        "text": "<email body of a legitimate banking notification>",
        "subject": "Step 4: Log in to Online Banking",
        "received_spf": "pass (spfCheck: domain of _spf.google.com designates ...)",
        "authentication_results": null,
        "reasons": [],
        "classification": "not_spam"
      },
      {
        "augmented": true,
        "text": "<email body of a legitimate identity verification TAN>",
        "subject": "Your WebID Solutions TAN",
        "received_spf": "pass (spfCheck: domain of _spf.google.com designates ...)",
        "authentication_results": "spf=pass ...; dmarc=fail header.from=icloud.com;",
        "reasons": [],
        "classification": "not_spam"
      }
      // ... 2 more demos (package tracking, flight gate notification)
    ],
    "signature": {
      "instructions": "... (see full instructions below)",
      "fields": [
        {
          "prefix": "Text:",
          "description": "The plaintext body of the email message to analyze"
        },
        {
          "prefix": "Subject:",
          "description": "The subject line key content of the email"
        },
        {
          "prefix": "Return Path:",
          "description": "The Return-Path header address"
        },
        {
          "prefix": "From Address:",
          "description": "The visible From header address"
        },
        {
          "prefix": "Received Spf:",
          "description": "The SPF verification result"
        },
        {
          "prefix": "Reply Address:",
          "description": "The Reply-To address if different from the sender"
        },
        {
          "prefix": "Authentication Results:",
          "description": "DKIM, SPF, DMARC results"
        },
        {
          "prefix": "Reasons:",
          "description": "Red flags justifying the classification"
        },
        {
          "prefix": "Classification:",
          "description": "The final spam risk level"
        }
      ]
    }
  }
}
```

The interesting parts are the `instructions` and the `demos`. The optimizer selected four demonstrations of legitimate emails (banking notifications, identity verification TANs, package tracking, flight gate changes). The full optimized instructions for the `classify` predictor:

```md
You are an expert cybersecurity email analyst specializing in spam and phishing detection. Analyze the provided email content and technical headers to classify the email's spam risk level.

CRITICAL CONTEXT: The recipient uses an email forwarding setup where legitimate emails are forwarded from Gmail to obfuscated iCloud.com addresses. This means:

- The From Address may show an iCloud.com domain even for legitimate senders (e.g., DHL, PayPal, Netflix, Lufthansa) due to the forwarding mechanism.
- DMARC authentication will frequently FAIL because the forwarding causes domain misalignment — this is EXPECTED and NOT a spam indicator in this context.
- The Return-Path often shows a Gmail address with '+caf\_=' forwarding notation, which is normal for this setup.
- SPF may pass for Google/Gmail infrastructure rather than the original sender's domain.

Do NOT flag these forwarding artifacts as spam indicators. Instead, focus on:

1. **Content analysis**: Does the email body contain phishing attempts, urgency manipulation, suspicious offers, requests for credentials, or deceptive language?
2. **Sender legitimacy**: Does the email content match what the claimed sender would actually send? (e.g., a DHL tracking notification with a real tracking number is legitimate)
3. **Contextual coherence**: Is the subject line consistent with the body? Does the email address the recipient by name? Is the language and formatting consistent with the claimed sender?
4. **True red flags**: Look for mismatched reply-to addresses pointing to unrelated domains, requests for sensitive information, suspicious attachments, or content that doesn't match the sender's typical communications.

Most emails in this system are legitimate German-language transactional emails. Classify conservatively — only escalate the risk level when there are genuine content-based or structural red flags beyond the expected forwarding artifacts. For legitimate emails, return an empty reasons list.
```

### What the Optimizer Found That I Wouldn't Have

Most of the generated instructions align with what I would have written manually. However, the optimizer explicitly identified that obfuscated `@icloud.com` addresses relayed via Google or Amazon with passing SPF checks should be treated as legitimate. It’s a perfect example of how data-driven optimization can surface subtle validation signals that are easily overlooked during manual prompt design.

### Trying Other LLMs

One of the biggest advantages of the DSPy approach is that switching models is trivial. Because the task is defined as a signature and a metric rather than a hand-tuned prompt, we can swap in a different LLM with a single line change and re-optimize, letting MIPROv2 find the best prompt for _that_ model's strengths.

Since we route everything through **OpenRouter**, trying a new model is as simple as changing the model identifier:

```python
import dspy

# Define candidate models to benchmark
models = {
    "gemini-2.5-flash":  "openrouter/google/gemini-2.5-flash",
    "gemini-3":          "openrouter/google/gemini-3",
    "grok-4.1-fast":     "openrouter/x-ai/grok-4.1-fast",
    "kimi-k2.5":         "openrouter/moonshotai/kimi-k2.5",
}

results = {}

for name, model_id in models.items():
    student_lm = dspy.LM(model_id)

    optimizer = dspy.MIPROv2(
        metric=spam_metric,
        auto="medium",
        prompt_model=teacher_lm,
        task_model=student_lm,
    )

    optimized = optimizer.compile(SpamClassifier(), trainset=trainset)

    evaluator = dspy.Evaluate(devset=testset, metric=spam_metric, num_threads=4)
    score = evaluator(optimized)
    results[name] = score

    # Save the optimized program for later use
    optimized.save(f"spam_classifier_optimized_{name}.json")
    print(f"{name}: {score:.4f}")
```

Each model gets its _own_ optimized instructions and demonstrations, tailored to its particular behavior. A prompt that works brilliantly for Gemini may be suboptimal for Grok, and MIPROv2 accounts for that automatically.

This also makes it easy to re-evaluate whenever a new model drops: add one entry to the `models` dict, run the loop, and compare scores on the same metric and test set.

## Conclusion

Programmatic optimization provides a systematic way to improve model performance by discovering patterns that are difficult to spot manually, such as the `@icloud.com` relay rule identified in this experiment.

The effectiveness of this approach relies heavily on a well-defined metric. By explicitly encoding the cost asymmetry between false positives and false negatives, the optimizer was forced to adopt a conservative classification strategy. This behavior emerged from the mathematical definition of success rather than from qualitative instructions.

Ultimately, this workflow decouples the problem definition from the implementation details. We define the signature, dataset, and metric, while the optimizer handles the prompt construction. This abstraction allows for reproducible optimizations and easier migration between different models as they become available.
