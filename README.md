# Pedestrian Positioning

This project is carried out in partnership with **IDEMIA** and **TELECOM Paris** as part of the IA Specialization program.


## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [License](#license)

## About

This project aims at creating an IA model to infer 2D pedestrian positions.
With this 2D position we should be capable to estimate depth and 
calculate a 3D position of said pedestrian.


### Prerequisites

- Python 3.11+
- `pip` package manager


### Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/JPGodTier/Pedestrian-Positioning-Idemia
    cd Pedestrian-Positioning-Idemia
    ```

2. **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Install the project in editable mode**:

    ```bash
    pip install -e .
    ```

## Configure GPU
To run the project using your GPU, please make sure you have a pytorch with CUDA
installed on your machine.

Install pytorch with CUDA: https://pytorch.org/get-started/locally/


## Dataset setup

### COCO
Go to https://cocodataset.org/#download

Download 2017 Train/Val annotations

## OCHumans

Go to https://github.com/liruilong940607/OCHumanApi?tab=readme-ov-file


## Usage

Here are the details on how to use each of the runner files located in the `bin` folder.

#### DataParser.py
`DataParser.py` prepares the data and puts it in a compatible training format (CSV).

**Usage**:
```bash
python bin/DataParser.py
````

#### RtmPoseParser.py
`RtmPoseParser.py` prepares the data and puts it in a compatible training format but infers keypoints position with Rtmpose instead of annotation files (like DataParser.py).

**Usage**:
```bash
python bin/RtmPoseParser.py
````

#### RunModel.py
`RunModel.py` runs the training process. Requires starting an MLflow server.
Note: Make sure the MLflow server is running before executing this script.

**Usage**:
```bash
mlflow ui
python bin/RunModel.py
````

#### VideoPositionRunner.py
`RunModel.py` runs the video processing pipeline that infers 2D and 3D positions of pedestrians.

**Usage**:
```bash
python bin/VideoPositionRunner.py
````

## License

Distributed under the MIT License. See `LICENSE` for more information.