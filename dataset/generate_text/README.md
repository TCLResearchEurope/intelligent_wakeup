# Conversational Data Generation

A flexible framework for generating conversational datasets using GPT (custom implementation similar to Beehive). This framework is designed to generate consistent, configurable conversations for various scenarios while allowing comparison between different LLM frameworks.

## Architecture Overview

- **Framework Agnostic**: Core logic is independent of specific LLM frameworks
- **Config-Driven**: Most behavior defined in configuration files
- **Modular**: Easy to add new frameworks or scenarios
- **Testable**: Unit tests for each component
- **Debug-Friendly**: Comprehensive logging and error handling

## Usage

### Basic Usage

```bash
python generate_text.py --config-dir config/ --output-dir output/
```

### Advanced Options

```bash
# Generate specific scenario
python generate_text.py --config-dir config/ --output-dir output/ --scenario smart_cooking

# Use specific framework
python generate_text.py --config-dir config/ --output-dir output/

# Generate multi-user conversations
python generate_text.py --config-dir config/ --output-dir output/ --conversation-type multi_user
```

### Command Line Arguments

- `--config-dir`: Directory containing configuration files (required)
- `--output-dir`: Directory for generated conversations (required)
- `--scenario`: Specific scenario to generate (or "all")
- `--conversation-type`: Type of conversation to generate

Here's the section you can add to your README explaining how to use apply_review.py:

## Dialogue Review & Enhancement

The framework includes a dialogue enhancement system that can improve the naturalness and authenticity of generated conversations.

### Using apply_review.py

The `apply_review.py` script allows you to apply dialogue enhancements to existing conversation files:

```bash
# Process a single file (will overwrite the original file)
python apply_review.py --input MovieRecommendation_0.json

# Process a single file with custom output location
python apply_review.py --input MovieRecommendation_0.json --output MovieRecommendation_0-enhanced.json

# Process all files in a directory
python apply_review.py --input dataset/ --output enhanced/ --batch
```

### Command Line Arguments

- `--input`: Input file or directory (required)
- `--output`: Output file or directory (defaults to overwriting input)
- `--config-dir`: Directory containing configuration files (default: 'config')
- `--rules`: Path to dialogue enhancement rules (default: 'config/postprocessing.json')
- `--scenario`: Path to specific scenario configuration file
- `--batch`: Process all files in input directory
- `--pattern`: File pattern to match in batch mode (default: '*.json')
- `--log-level`: Set logging verbosity ('debug', 'info', 'warning', 'error', 'critical')

## Configuration

### Framework Configuration (config/frameworks.json)
```json
{
    "custom": {
        "description": "Framework configuration",
        "supported_models": ["gpt-4", "gpt-3.5-turbo"],
        "default_model": "gpt-3.5-turbo"
    }
}
```

### Scenario Configuration (config/scenarios/smart_cooking.json)
```json
{
    "name": "SmartCooking",
    "framework_config": {
        "system_prompt": "You are generating cooking assistance conversations.",
        "chat_loops": 4,
        "model_config": {
            "name": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    },
    "variations": [
        {
            "context": "Basic recipe help",
            "complexity": "simple"
        }
    ]
}
```

## Adding a New Scenario

1. Create scenario configuration in `config/scenarios/`
2. Define variations and agent configurations
3. Add framework-specific settings

## Output Format

Generated conversations are saved in JSON format:

```json
{
    "scenario_type": "SmartCooking",
    "conversation_type": "single_user",
    "context": "Basic recipe help",
    "conversation": [
        {
            "speaker": "User",
            "content": "How do I make pasta al dente?",
            "timestamp": "2024-12-09T10:30:00"
        }
    ]
}
```
