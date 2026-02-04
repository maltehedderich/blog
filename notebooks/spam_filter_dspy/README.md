# DSPy Spam Filter Notebook

This notebook builds and optimizes a DSPy-based spam classifier using a Phoenix dataset and OpenRouter language models. It instruments DSPy with OpenTelemetry so traces can be exported to Phoenix for observability.

## What the notebook does

- Connects to a Phoenix instance and loads the `spam-classification` dataset.
- Configures OpenTelemetry tracing and instruments DSPy.
- Sets up OpenRouter-backed DSPy language models for task and prompt optimization.
- Defines a `SpamClassification` signature and `SpamClassifier` module.
- Prepares a DSPy training set from Phoenix examples.
- Implements a custom metric with asymmetric penalties.
- Optimizes the classifier with `dspy.MIPROv2`.

## Prerequisites

- Access to a Phoenix instance (OTLP endpoint + API key).
- An OpenRouter API key.
- Python environment with required packages (Phoenix client, OpenInference DSPy instrumentation, OpenTelemetry, DSPy).

## Running the notebook

1. Open [notebooks/spam_filter_dspy/spam_filter.ipynb](notebooks/spam_filter_dspy/spam_filter.ipynb).
2. Run the cells in order.
3. When prompted, provide:
   - Phoenix OTLP endpoint (e.g., `https://your-phoenix.com`)
   - Phoenix API key
   - OpenRouter API key

## Notes

- The metric treats false positives (ham marked as spam) as most costly.
- The optimization step may take time depending on dataset size and model latency.
- You can disable instrumentation by running the provided uninstrument cell if needed.

## Outputs

- An optimized `SpamClassifier` instance stored in `optimized`.
- Trace data emitted to Phoenix for each DSPy call.
