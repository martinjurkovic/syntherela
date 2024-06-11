## Install benchmark reproducibility environment
```bash
conda create -n syntherela python=3.10 -y
conda activate reproduce_benchmark
pip install -r requirements.txt
```

## Install tabular
```bash
conda create -n tabular python=3.9 -y
conda activate tabular
pip install .
pip install -r experiments/generation/tabular/requirements.txt
```

## Install rctgan
```bash
conda create -n rctgan python=3.7 -y
conda activate rctgan
pip install experiments/generation/rctgan/RCTGAN 
```

## Install gretel
```bash
conda create -n gretel python=3.9 -y
conda activate gretel
pip install .
pip install -r experiments/generation/gretel/requirements.txt
```