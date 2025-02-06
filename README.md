
---

# Data Fitting Group Project: Approximating an Unknown Function

## Overview

This project aims to approximate an unknown function $ f: \mathbb{R}^2 \to \mathbb{R} $ using various data fitting techniques. The implemented methods include:

1. **Linear Regression**
2. **Polynomial Regression**
3. **Neural Networks**

The program evaluates the performance of each technique using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Coefficient of Determination ($ R^2 $).

---

## Table of Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
  - [Creating a Virtual Environment](#creating-a-virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Running the Program](#running-the-program)
- [File Structure](#file-structure)

---

## Setup Instructions

### Creating a Virtual Environment

To ensure that all dependencies are isolated and do not conflict with other projects, it is recommended to use a Python virtual environment. Follow these steps to create and activate a virtual environment:

#### On macOS/Linux:

```bash
python3 -m venv myenv
source myenv/bin/activate
```

#### On Windows:

```bash
python -m venv myenv
myenv\Scripts\activate
```

After activation, you should see `(myenv)` in your terminal prompt, indicating that the virtual environment is active.

---

### Installing Dependencies

Once the virtual environment is activated, install the required libraries by running:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages listed in the `requirements.txt` file, including:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `tensorflow`

You can verify the installed packages by running:

```bash
pip list
```

---

## Running the Program

To run the program, execute the following command:

```bash
python main.py
```

Ensure that your dataset (`data.txt`) is in the same directory as the script or update the `file_path` variable in the code to point to your dataset.

The program will:

1. Load the dataset.
2. Train and evaluate three models: Linear Regression, Polynomial Regression, and Neural Networks.
3. Print the performance metrics (MSE, MAE, $ R^2 $) for each model.

---

## File Structure

The project has the following structure:

```
project/
│
├── main.py               # Main script containing the implementation
├── data.txt              # Dataset file with inputs (x, y) and outputs f(x, y)
├── requirements.txt      # List of required Python libraries
└── README.md             # This file
```

---

## Additional Notes

- **Virtual Environment Deactivation**: To deactivate the virtual environment after finishing your work, simply run:

  ```bash
  deactivate
  ```

- **Deleting the Virtual Environment**: If you no longer need the virtual environment, you can delete the `myenv` folder:

  ```bash
  rm -rf myenv  # macOS/Linux
  rmdir /s myenv  # Windows
  ```

- **Customizing the Dataset**: Replace `data.txt` with your own dataset, ensuring it follows the same format (`x, y, f(x, y)`).

---

## Future Work

Potential directions for future work include:

- Exploring additional data fitting techniques (e.g., Gaussian Processes, Support Vector Machines).
- Evaluating model performance on larger or more complex datasets.
- Investigating the impact of regularization techniques on model performance.

---

Feel free to modify this `README.md` file to better suit your project's specific details!
