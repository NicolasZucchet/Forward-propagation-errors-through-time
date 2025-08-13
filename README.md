# Online-BPTT

This repository contains the code for the paper "Online Backpropagation-Through-Time".

## Setup

1. Install uv
    ```bash
    pip install uv
    ```
2. Create a virtual environment
    ```bash
    uv venv env
    ```
3. Activate the virtual env
    ```bash
    source env/bin/activate
    ```
4. Install the dependencies
    ```bash
    uv pip install -e .
    ```

## Running Tests

Run pytest from the root of the project:
```bash
uv run pytest
```
