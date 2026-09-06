# Intelligent Wakeup: Dataset

This repository contains a pipeline for generating dataset used in testing intelligent wakeup systems. The pipeline creates realistic multi-party conversations with varying scenarios and background conditions, synthesizes them into speech, and generates final audio scenes.

The dataset generation process consists of three main steps:
1. **Text Generation**: Creates natural conversations using LLM agents
2. **Speech Synthesis**: Converts generated text into speech using OpenVoice v2 and ElevenLabs API
3. **Audio Scene Creation**: Combines speech with background noise and effects

## Generate Dataset

### Prerequisites
- Anaconda or Miniconda installed
- OpenAI API key (for text generation)

### Installation

Activate project enviroment (if you haven't created it then: `conda env create -f ../enviroment.yml`):
1. Create and activate conda environment:
```bash
conda activate intelligent_wakeup
```

2. Install dependencies:
```bash
bash install_dependencies.sh
```

3. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

### Generate Corpus

To generate the complete dataset:
```bash
./generate_dataset.sh
```

For individual steps:
```bash
cd generate_text && ./build.sh         # Generate conversation texts
cd generate_speech && ./build.sh       # Synthesize speech using OpenVoice
cd generate_audio_scenes && ./build.sh # Create final audio scenes
```

## Citation

If you use this corpus, please cite:
```
@inproceedings{sowanski2024intelligentwakeup,
  title={Intelligent Wakeup for Seamless Virtual Assistant},
  author={Sowa{\'n}ski, Marcin and Palich, Cezary and Fraszczak, Jakub},
  booktitle={},
  pages={},
  year={2025},
  organization={}
}
```
