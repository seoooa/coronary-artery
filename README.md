# Coronary Artery Segmentation
A deep learning-based framework for coronary artery segmentation. This project accurately segments coronary arteries from medical images to support cardiovascular disease diagnosis and treatment planning.

## Project Structure

The project is organized as follows:
- `data/`: Directory for storing datasets
- `nbs/`: Directory for Jupyter Notebook files
- `script/`: Directory for experiment and test scripts
  - `unit_test/`: Directory for unit test scripts
- `src/`: Directory for source code
  - `archs/`: Code related to model architectures
  - `data/`: Code for data processing
  - `losses/`: Code for loss functions
  - `metrics/`: Code for evaluation metrics
  - `models/`: Code for model implementations
  - `utils/`: Directory for utility functions
- `script/`: Scripts for running the demo

## Installation

### Prerequisites
- Python 3.10
- uv (for dependency management and virtual environment)

### Setup

#### Option 1: Using uv (Recommended)

1. Install uv if you don't have it already:
   ```
   pip install uv
   ```

2. Clone the repository:
   ```
   git clone https://github.com/seoooa/coronary-artery.git
   cd coronary-artery
   ```

3. Create a virtual environment and install dependencies using uv:
   ```
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

4. Install dependencies:
   ```
   uv pip install -r requirements.txt
   ```

5. Install PyTorch with CUDA support:
   ```
   uv pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

#### Option 2: Using pipenv (Legacy)

1. Install dependencies using pipenv:
   ```
   pipenv install
   ```

2. Activate the virtual environment:
   ```
   pipenv shell
   ```