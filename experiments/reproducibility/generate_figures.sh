#!/bin/bash

mkdir -p results/figures
echo "Drawing figure 1"
python experiments/figures/figure1.py
echo "Drawing figure 2"
python experiments/figures/figure2.py
echo "Drawing figure 3"
python experiments/figures/figure3.py
echo "Drawing figure 4"
python experiments/figures/figure4.py
echo "Drawing figure 5"
python experiments/figures/figure5.py
echo "Appendix figures (6, 7) might take a while to generate"
echo "Drawing figure 6"
python experiments/figures/figure6.py
echo "Drawing figure 7"
python experiments/figures/figure7.py
echo "Figures can be found in results/figures"
