# Bayesian Regression with Pyro

This Python code demonstrates Bayesian regression using the Pyro library. Bayesian regression is a powerful technique for modeling relationships between variables while accounting for uncertainty in model parameters. The code covers data generation, defining a Bayesian regression model, performing Bayesian inference using Stochastic Variational Inference (SVI), and visualizing posterior distributions of the model parameters.

## **Theory of Bayesian Linear Regression**

### **Introduction**

Linear regression is a popular regression approach in machine learning. It is based on the assumption that the underlying data is normally distributed and that all relevant predictor variables have a linear relationship with the outcome. However, in the real world, this assumption may not always hold, making Bayesian regression a better choice. Bayesian regression leverages prior beliefs or knowledge about the data to "learn" more about it and make more accurate predictions. It also accounts for data uncertainty and uses prior knowledge to provide more accurate estimates.

### **Bayesian Regression**

Bayesian regression is a type of linear regression that uses Bayesian statistics to estimate unknown parameters of the model. It uses Bayes' theorem to estimate the probability of a set of parameters given the observed data. The goal of Bayesian regression is to find the best estimate of the parameters of the linear model that describes the relationship between independent and dependent variables.

### **Some Dependent Concepts for Bayesian Regression**

**Bayes' Theorem**

Bayes' theorem provides a relationship between the prior probability of an event and its posterior probability after considering the evidence. It states that the conditional probability of an event is equal to the likelihood of the event given certain conditions multiplied by the prior probability of the event, divided by the probability of the conditions.

**Maximum Likelihood Estimation (MLE)**

MLE is a method used to estimate the parameters of a statistical model by maximizing the likelihood function. It aims to find parameter values that make the observed data most probable within the assumed model.

**Maximum A Posteriori (MAP) Estimation**

MAP estimation is a Bayesian approach that combines prior information with the likelihood function to estimate parameters. It involves finding parameter values that maximize the posterior distribution, obtained by applying Bayes' theorem.

### **Implementation of Bayesian Regression**

Let's implement a Bayesian regression model using the Pyro library.

## Installation

To run this code, you need to install the Pyro library. You can install it using the following command:

```bash
pip install pyro-ppl
```

## Code Overview

### 1. Data Generation

Simulated data with a linear relationship and random noise is generated.

```python
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import Predictive

true_slope = 2
true_intercept = 1

X = torch.linspace(0, 10, 100)
Y = true_intercept + true_slope * X + torch.randn(100)
```

### 2. Bayesian Regression Model

A Bayesian regression model is defined with prior distributions for the slope, intercept, and standard deviation. The likelihood is modeled using a normal distribution.

```python
def model(X, Y):
    slope = pyro.sample("slope", dist.Normal(0, 10))
    intercept = pyro.sample("intercept", dist.Normal(0, 10))
    sigma = pyro.sample("sigma", dist.HalfNormal(1))

    mu = intercept + slope * X

    with pyro.plate("data", len(X)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=Y)
```

### 3. Bayesian Inference Using SVI

Stochastic Variational Inference (SVI) is used for Bayesian inference. A guide function is defined to approximate the posterior distributions of the model parameters.

```python
def guide(X, Y):
    slope_loc = pyro.param("slope_loc", torch.tensor(0.0))
    slope_scale = pyro.param("slope_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    intercept_loc = pyro.param("intercept_loc", torch.tensor(0.0))
    intercept_scale = pyro.param("intercept_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    sigma_loc = pyro.param("sigma_loc", torch.tensor(1.0), constraint=dist.constraints.positive)

    slope = pyro.sample("slope", dist.Normal(slope_loc, slope_scale))
    intercept = pyro.sample("intercept", dist.Normal(intercept_loc, intercept_scale))
    sigma = pyro.sample("sigma", dist.HalfNormal(sigma_loc))
```

### 4. Training the Model

SVI optimization is performed to train the Bayesian regression model.

```python
optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

num_iterations = 1000

for i in range(num_iterations):
    loss = svi.step(X, Y)
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}/{num_iterations} - Loss: {loss}")
```

### 5. Posterior Samples

Posterior samples are obtained using the `Predictive` module.

```python
predictive = Predictive(model, guide=guide, num_samples=1000)
posterior = predictive(X, Y)

slope_samples = posterior["slope"]
intercept_samples = posterior["intercept"]
sigma_samples = posterior["sigma"]
```

### 6. Parameter Estimation

Mean values of the posterior samples are computed to estimate the parameters.

```python
slope_mean = slope_samples.mean()
intercept_mean = intercept_samples.mean()
sigma_mean = sigma_samples.mean()

print("Estimated Slope:", slope_mean.item())
print("Estimated Intercept:", intercept_mean.item())
print("Estimated Sigma:", sigma_mean.item())
```

### 7. Visualization of Posterior Distributions

Posterior distributions of the slope, intercept, and standard deviation are visualized using kernel density plots.

```python
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.kdeplot(slope_samples, shade=True, ax=axs[0])
axs[0].set_title("Posterior Distribution of Slope")
axs[0].set_xlabel("Slope")
axs[0].set_ylabel("Density")

sns.kdeplot(intercept_samples, shade=True, ax=axs[1])
axs[1].set_title("Posterior Distribution of Intercept")
axs[1].set_xlabel("Intercept")
axs[1].set_ylabel("Density")

sns.kdeplot(sigma_samples, shade=True, ax=axs[2])
axs[2].set_title("Posterior Distribution of Sigma")
axs[2].set_xlabel("Sigma")
axs[2].set_ylabel("Density")

plt.tight_layout()
plt.show()
```

![Posterior Distributions](https://github.com/denis-samatov/Bayesian_regression_model/blob/main/img.png)

## Results

### Advantages of Bayesian Regression:

- **Effective with small data sizes:** Bayesian regression is very effective when there is limited data.
- **Suitable for online learning:** Especially useful for online learning, where data arrives in real-time, compared to batch learning.
- **Mathematically robust:** The Bayesian approach is a proven and mathematically sound method that can be used even without prior knowledge of the data.
- **Ability to incorporate external information:** Bayesian methods use priors, allowing the incorporation of external information into the model.

### Disadvantages of Bayesian Regression:

- **Inference time:** Inference can take a long time.
- **Inefficiency with large data sizes:** For large datasets, the Bayesian approach can be less efficient compared to frequentist methods.
- **Installation issues:** If installing new packages is difficult, it can be a problem.
- **Dependence on linearity:** Bayesian models are also susceptible to errors inherent in traditional frequentist models, and they still rely on linear relationships between features and the outcome variable.

### When to Use Bayesian Regression:

- **Small sample sizes:** Bayesian inference is particularly useful when dealing with small data sizes. It is a good choice when needing to develop a complex model but with limited data.
- **Reliable prior knowledge:** A straightforward way to incorporate reliable external knowledge into a model is by using a Bayesian model. The impact of priors will be more pronounced when working with small datasets.
