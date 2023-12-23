# Piece of code for structural topology optimization. 

```SIMP_code.py``` contains a python code for conducting structural topology 
optimization using the Solid Isotropic Material with Penalization (SIMP) method.

![Optimized Structure](images/density_field.png)
 
The SIMP method is a widely used approach to find the optimal distribution of 
material within a design domain, aiming to minimize compliance (maximize 
stiffness) under given constraints.

## Features

- Define and customize design domain and boundary conditions.
- Specify material properties and constraints.

## Prerequisites

Before using this code, you'll need to have the following installed:

- Python 3
- Required Python libraries (```numPy``` and ```fenicsx```)

## Usage

```sh
python SIMP_code.py
```