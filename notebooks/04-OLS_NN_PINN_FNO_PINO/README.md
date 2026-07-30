# OLS, Neural Networks, PINNs, FNOs and PINOs for Materials Science (Updated notebook)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sai-mat-group/ai-ml-for-materials-2026/blob/main/notebooks/04-OLS_NN_PINN_FNO_PINO/04-updated_OLS_NN_PINN_FNO_PINO.ipynb)

This notebook develops a continuous teaching sequence from ordinary least squares to neural networks, physics-informed neural networks, Fourier neural operators and physics-informed neural operators. The examples are written for materials-science learners and emphasize a central principle:

> **Fitting is a mathematical operation. Physical reasoning determines the model, the variables, the admissible domain and the interpretation of deviations from the fit.**

## Notebook

- [`OLS_NN_PINN_FNO_PINO.ipynb`](./04-updated_OLS_NN_PINN_FNO_PINO.ipynb)

The notebook is designed for Google Colab and can also be executed locally in Jupyter.

## Learning path

### Ordinary least squares and the Hall–Petch relation

The notebook begins with one synthetic grain-size–strength dataset:

- 10 inverse Hall–Petch observations between 1 and 10 nm;
- 500 ordinary Hall–Petch observations between 10 and 500 nm;
- one continuous strength maximum at the transition grain size of 10 nm.

The dominant ordinary Hall–Petch population is used to define the train–validation–test split and to estimate

$$\sigma_y=\sigma_0+k d^{-1/2} $$

The ten small-grain observations are initially tagged as anomalous scatter. They remain visible in the common dataset but are excluded from the ordinary OLS fit. This deliberately illustrates how a numerically excellent fit can be physically incomplete when a small, systematic regime is neglected.

The notebook includes:

- yield strength versus grain size \(d\), with a logarithmic grain-size axis;
- yield strength versus \(d^{-1/2}\), the conventional linear Hall–Petch coordinate;
- residual diagnostics;
- a deliberately restricted analysis that hides the inverse Hall–Petch observations;
- recovery of the same Hall–Petch parameters using gradient-based optimization.

### Neural networks on the same dataset

The neural-network section does not generate a second physical dataset. It reuses the same 510 observations.

The ten inverse Hall–Petch observations are added to the neural-network training set, while validation and test data remain drawn from the dominant ordinary Hall–Petch population. Underfitting, balanced fitting and overfitting are compared using common scaling and common data.

### Bias and variance

The bias–variance demonstration also preserves the same physical dataset and grain-size coordinates. Repeated experiments differ only through new realizations of measurement noise.

Polynomial models of different flexibility are compared through:

- mean prediction;
- prediction spread;
- squared bias;
- variance;
- separate metrics for the ordinary and inverse Hall–Petch regimes.

This avoids changing the constitutive relation merely to demonstrate model complexity.

### Forward PINN

A forward physics-informed neural network solves a one-dimensional Poisson problem using:

- equation residuals at collocation points;
- boundary-condition losses;
- validation residuals at independent coordinates;
- comparison with a reference solution that is not used as interior training data.

### Staged inverse PINN for binary diffusion

The inverse-PINN section is divided into short, unnumbered stages with Markdown explanations between executable cells.

It introduces:

- a composition-dependent interdiffusion coefficient;
- a synthetic diffusion-couple profile;
- the Matano plane and Boltzmann coordinate;
- a neural representation of the composition profile;
- the Boltzmann-transformed diffusion residual;
- profile pretraining;
- constitutive parameter estimation.

Two inverse situations are compared directly.

**Composition profile and physics only**

The model receives:

- one measured composition profile;
- terminal compositions;
- the governing diffusion equation;
- an assumed Arrhenius–quadratic constitutive family.

A small profile error does not necessarily imply unique recovery of the diffusivity.

**Composition profile, physics and sparse diffusivity data**

The same inverse problem is supplemented by three independent diffusivity measurements at selected compositions. The added local constitutive information generally improves recovery of the complete diffusivity curve.

The comparison demonstrates that reproducing a concentration profile and identifying the underlying constitutive law are distinct inverse problems.

### FNO and PINO

The final sections distinguish among:

- a neural network that approximates one function;
- a PINN that solves one differential-equation instance;
- an FNO that learns a map between input and solution fields from solved examples;
- a PINO that combines operator learning with equation-based diagnostics or training;
- operator-assisted inversion in which a trained operator is held fixed while unknown input parameters are optimized.

## Data-integrity rule used throughout

The Hall–Petch and neural-network sections use one common dataset. No later section silently changes:

- the grain-size range;
- the number of physical observations;
- the transition grain size;
- the constitutive relation;
- the train–validation–test indices.

Repeated-sampling experiments change only the random noise realization.

## Running in Google Colab

Open the notebook using the badge at the top of this page and select:

`Runtime -> Run all`

Run cells in order because later cells use objects defined earlier.

A GPU is optional. The notebook selects an available device automatically.

## Running locally

A recent Python environment with the following packages is sufficient:

```bash
python -m pip install numpy matplotlib torch jupyter
```

Then start Jupyter:

```bash
jupyter notebook notebooks/04-OLS_NN_PINN_FNO_PINO/04-updated_OLS_NN_PINN_FNO_PINO.ipynb
```

## Main dependencies

- Python
- NumPy
- Matplotlib
- PyTorch
- Jupyter or Google Colab

## Interpretation cautions

- A low RMSE does not establish that the model contains the complete physics.
- A high \(R^2\) applies only to the observations and domain used to compute it.
- A systematic group of apparent outliers may indicate a second physical regime.
- A fitted concentration profile is not automatically a pointwise diffusivity measurement.
- Sparse constitutive measurements can materially improve inverse identifiability.
- Loss weights are optimization choices, not material constants.
- Synthetic truth is used for verification only and would not be available in an experimental inverse problem.

## Historical Hall–Petch references

The notebook uses the classic Hall–Petch relation as an example of physics-guided model construction. In the original works, the physical interpretation of deformation, grain boundaries and fracture motivated the relation; fitting followed the physical argument.

- E. O. Hall, “The Deformation and Ageing of Mild Steel: III Discussion of Results,” *Proceedings of the Physical Society. Section B*, **64** (1951) 747–753. DOI: `10.1088/0370-1301/64/9/303`.
- N. J. Petch, “The Cleavage Strength of Polycrystals,” *Journal of the Iron and Steel Institute*, **174** (1953) 25–28.

### To note

The notebook should not be presented as a progression in which increasingly flexible fitting automatically reveals physics. Instead, it should be presented as a comparison of mathematical tools under explicit physical assumptions:

- physics motivates the coordinates and model class;
- fitting estimates parameters within that class;
- residuals and excluded observations test the model’s domain of validity;
- flexible models can expose structure but do not independently prove a mechanism;
- inverse problems require attention to identifiability and independent information.
