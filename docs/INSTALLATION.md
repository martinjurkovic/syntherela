## Install benchmark reproducibility environment
```bash
conda create -n reproduce_benchmark python=3.10 -y
conda activate reproduce_benchmark
pip install -r requirements.txt
```

## Install tabular
```bash
conda create -n tabular python=3.9 -y
conda activate tabular
pip install syntherela
pip install -r experiments/generation/tabular/requirements.txt
```

## Install rctgan
```bash
conda create -n rctgan python=3.7 -y
conda activate rctgan
git clone https://github.com/ValterH/RCTGAN.git experiments/generation/rctgan/RCTGAN
pip install experiments/generation/rctgan/RCTGAN 
```

## Install realtabformer
```bash
conda create -n realtabformer python=3.9 -y
conda activate realtabformer
pip install syntherela
pip install -r experiments/generation/realtabformer/requirements.txt
```

## Install gretel
```bash
conda create -n gretel python=3.9 -y
conda activate gretel
pip install syntherela
pip install -r experiments/generation/gretel/requirements.txt
```

## Install mostlyai
```bash
conda create -n mostlyai python=3.10 -y
conda activate mostlyai
pip install syntherela
pip install -r experiments/generation/mostlyai/requirements.txt
```

## Install ClavaDDPM
```bash
conda create -n clavaddpm python=3.9 -y
conda activate clavaddpm
git clone https://github.com/weipang142857/ClavaDDPM.git experiments/generation/clavaddpm/ClavaDDPM
pip install -r experiments/generation/clavaddpm/requirements.txt
```
