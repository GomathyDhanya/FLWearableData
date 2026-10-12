# Federated Learning Assignment

Implementation for the federated learning assignment using the UCI Localization Data for Person Activity dataset.

## Project Structure

```text
fedavg_minimal/
├── config.py
├── data.py
├── model.py
├── fedavg.py
├── mia.py
├── privacy.py
├── plots.py
├── main.py
└── requirements.txt
```

## Run

Unzip the project and enter the folder:

```bash
unzip fedavg_minimal.zip
cd fedavg_minimal
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```



Run the assignment:

```bash
python main.py
```

## What It Runs

The script will:

- download and preprocess the UCI dataset
- build the 5-class temporal-window dataset
- run all Task A FedAvg configurations
- evaluate Round-1 LOSS membership inference for Task B
- run the DP-SGD + CKKS privacy extension
- generate plots and result files

Outputs are saved in:

```text
Final_Results/
```

## Notes

The full experiment may take some time because multiple FedAvg configurations are trained, with a maximum of 300 communication rounds per configuration.
