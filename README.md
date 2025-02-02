# AiFlow
AiFlow is a lightweight Python package that provides a simple and easy-to-use interface for building, evaluating, and training machine learning models. It is designed to quickly build and train models without adding the overhead of complex frameworks. With integrated support for LoRA enhancements, audio transcription, web scraping, and various text generation techniques, AiFlow offers a versatile toolkit for both research and production environments.

## Table of Content
- [AiFlow](#aiflow)
  - [Table of Content](#table-of-content)
  - [Features](#features)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Documentation \& Usage](#documentation--usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## Features
- **Model Training & Inference:**  
  Train state-of-the-art language models with support for LoRA. The [`YunaLLMTrainer`](aiflow/trainer.py) class provides methods for creating, training, and merging models.
  
- **Audio Data Processing:**  
  Easily combine, split, and process audio files using the [`AudioDataWorker`](aiflow/data.py) class. Perform transcription with integrated ASR pipelines.
  
- **Text Data Processing:**  
  Process and split large text files, count tokens, and convert between different formats using the [`TextDataWorker`](aiflow/data.py) class.
  
- **Web & HTML scraping:**  
  Retrieve and clear web data for quick processing using helper functions from [`helper.py`](aiflow/helper.py).
  
- **Conditional Import Management:**  
  Depending on configuration settings, AiFlow loads the necessary modules for tasks such as image model loading, voice synthesis, or text generation using advanced tokenization and LoRA techniques.
  
- **Extensible & Configurable:**  
  Customize your project with the provided configuration system. Use the [`get_config`](aiflow/helper.py) function to load and manage settings.

## Installation
To install AiFlow, follow the steps below:

1. Build the package:
    ```bash
    python setup.py sdist bdist_wheel
    ```
2. Install the package from the generated wheel:
    ```bash
    pip install dist/aiflow-2.0.0-py3-none-any.whl --force-reinstall
    ```

## Quick Start
You can train an LLM model using Colab Notebook provided in the repository.

## Documentation & Usage
Each module in AiFlow is fully documented with inline comments. Refer to file documentation for detailed usage:
- Trainer Module
- Helper Utilities
- AGI Module
- Data Processing

## Contributing
Contributions are welcome! For feature requests, bug reports, or other issues, please open an issue in the repository. If you would like to contribute code, please fork the repository and submit a pull request.

## License
AiFlow is distributed under the OSI Approved Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License.

## Contact
For questions or support, please open an issue in the repository or contact the author at yukiarimo@gmail.com.