# Heartly: Heart Disease Risk Assessment App

![Heartly Logo](https://github.com/Kawai-Senpai/Heartly/blob/381aaaf70745ef58905fa69037e4af89acd513c1/Heartly_LOGO.png)

## Overview

**Heartly** is a Python application developed with Kivy, designed to evaluate the risk of heart disease based on user inputs developed by [*Ranit Bhowmick*](https://www.linkedin.com/in/ranitbhowmick/) & [*Sayanti Chatterjee*](https://www.linkedin.com/in/sayantichatterjee/). The application utilizes a neural network model to predict the likelihood of heart disease, providing a percentage risk based on a series of questions related to the user's health and lifestyle.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Features](#features)
4. [Usage](#usage)
5. [Technical Details](#technical-details)
   - [Neural Network Model](#neural-network-model)
   - [Data Collection](#data-collection)
   - [Risk Calculation](#risk-calculation)
   - [UI/UX Design](#uiux-design)
   - [Error Handling](#error-handling)
6. [Testing](#testing)
7. [Acknowledgements](#acknowledgements)

## Installation

To set up Heartly on your local machine, follow these instructions:

### Prerequisites

- Python 3.7 or higher
- Kivy 2.0.0 or higher
- PyTorch 1.10.0 or higher
- Pip (Python Package Installer)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kawai-Senpai/Heartly.git
cd Heartly
```

### Step 2: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python kivy_Chat_HeartDisease.py
```

This will start the Heartly application. Follow the on-screen instructions to complete the risk assessment.

## Features

- **Neural Network Prediction:** Utilizes a deep neural network to calculate the risk of heart disease.
- **Interactive Questionnaire:** Engages users with a series of health-related questions.
- **Real-Time Risk Assessment:** Displays the risk percentage in real-time as users answer the questions.
- **Cross-Platform Compatibility:** Compatible with Windows, macOS, and Linux.


<div style="display: flex; gap: 10px; flex-wrap: wrap;">

<img src="https://github.com/Kawai-Senpai/Heartly/blob/7bfcee9326100bc60ddd8ae0c931a6f0abe1f14d/Screenshots/Heartly%20Screenshot%20(1).png" alt="Screenshot 1" width="200"/>
<img src="https://github.com/Kawai-Senpai/Heartly/blob/7bfcee9326100bc60ddd8ae0c931a6f0abe1f14d/Screenshots/Heartly%20Screenshot%20(2).png" alt="Screenshot 2" width="200"/>
<img src="https://github.com/Kawai-Senpai/Heartly/blob/7bfcee9326100bc60ddd8ae0c931a6f0abe1f14d/Screenshots/Heartly%20Screenshot%20(3).png" alt="Screenshot 3" width="200"/>
<img src="https://github.com/Kawai-Senpai/Heartly/blob/7bfcee9326100bc60ddd8ae0c931a6f0abe1f14d/Screenshots/Heartly%20Screenshot%20(4).png" alt="Screenshot 4" width="200"/>
<img src="https://github.com/Kawai-Senpai/Heartly/blob/7bfcee9326100bc60ddd8ae0c931a6f0abe1f14d/Screenshots/Heartly%20Screenshot%20(5).png" alt="Screenshot 5" width="200"/>

</div>



## Usage

Upon launching Heartly, users will be presented with a questionnaire covering various health and lifestyle factors. Each response is processed by the neural network model to estimate the risk of heart disease. The application then displays the calculated risk percentage based on the provided answers.

## Technical Details

### Neural Network Model

The core of Heartly's prediction capability is a neural network model implemented using PyTorch. The model consists of 23 fully connected layers with GELU activation functions and dropout for regularization. The architecture is as follows:

- **Layers:**
  - `fc1` to `fc23`: Fully connected layers with decreasing output sizes from 40 to 1.
  - `af` (Activation Function): GELU (Gaussian Error Linear Unit).
  - `sigmoid`: Sigmoid activation function applied to the final layer's output.
  - `dropout`: Dropout layer with a dropout rate of 0.05.

**Model Description:**

```python

#Copy of the net
class neuralnet(nn.Module):

    """
    Neural network model for heart disease prediction.
    Attributes:
        fc1 (nn.Linear): Fully connected layer with input size 18 and output size 40.
        fc2 (nn.Linear): Fully connected layer with input size 40 and output size 50.
        fc3 (nn.Linear): Fully connected layer with input size 50 and output size 60.
        fc4 (nn.Linear): Fully connected layer with input size 60 and output size 70.
        fc5 (nn.Linear): Fully connected layer with input size 70 and output size 80.
        fc6 (nn.Linear): Fully connected layer with input size 80 and output size 90.
        fc7 (nn.Linear): Fully connected layer with input size 90 and output size 100.
        fc8 (nn.Linear): Fully connected layer with input size 100 and output size 110.
        fc9 (nn.Linear): Fully connected layer with input size 110 and output size 120.
        fc10 (nn.Linear): Fully connected layer with input size 120 and output size 130.
        fc11 (nn.Linear): Fully connected layer with input size 130 and output size 120.
        fc12 (nn.Linear): Fully connected layer with input size 120 and output size 110.
        fc13 (nn.Linear): Fully connected layer with input size 110 and output size 100.
        fc14 (nn.Linear): Fully connected layer with input size 100 and output size 90.
        fc15 (nn.Linear): Fully connected layer with input size 90 and output size 80.
        fc16 (nn.Linear): Fully connected layer with input size 80 and output size 70.
        fc17 (nn.Linear): Fully connected layer with input size 70 and output size 60.
        fc18 (nn.Linear): Fully connected layer with input size 60 and output size 50.
        fc19 (nn.Linear): Fully connected layer with input size 50 and output size 40.
        fc20 (nn.Linear): Fully connected layer with input size 40 and output size 30.
        fc21 (nn.Linear): Fully connected layer with input size 30 and output size 20.
        fc22 (nn.Linear): Fully connected layer with input size 20 and output size 10.
        fc23 (nn.Linear): Fully connected layer with input size 10 and output size 1.
        af (nn.GELU): Activation function GELU.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
        dropout (nn.Dropout): Dropout layer with dropout rate of 0.05.
    """
    """
    Forward pass of the neural network.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Output tensor after passing through the network.
    """

    def __init__(self):
        super(neuralnet,self).__init__()

        self.fc1=nn.Linear(18,40)
        self.fc2=nn.Linear(40,50)
        self.fc3=nn.Linear(50,60)
        self.fc4=nn.Linear(60,70)
        self.fc5=nn.Linear(70,80)
        self.fc6=nn.Linear(80,90)
        self.fc7=nn.Linear(90,100)
        self.fc8=nn.Linear(100,110)
        self.fc9=nn.Linear(110,120)
        self.fc10=nn.Linear(120,130)
        self.fc11=nn.Linear(130,120)
        self.fc12=nn.Linear(120,110)
        self.fc13=nn.Linear(110,100)
        self.fc14=nn.Linear(100,90)
        self.fc15=nn.Linear(90,80)
        self.fc16=nn.Linear(80,70)
        self.fc17=nn.Linear(70,60)
        self.fc18=nn.Linear(60,50)
        self.fc19=nn.Linear(50,40)
        self.fc20=nn.Linear(40,30)
        self.fc21=nn.Linear(30,20)
        self.fc22=nn.Linear(20,10)
        self.fc23=nn.Linear(10,1)
        self.af=nn.GELU()
        self.sigmoid=nn.Sigmoid()
        self.dropout = nn.Dropout(0.05)

    def forward(self,x):
        
        out = self.fc1(x)
        out = self.af(out)
        out = self.fc2(out)
        out = self.af(out)
        out = self.fc3(out)

        out = self.fc4(out)
        out = self.af(out)
        out = self.fc5(out)
        out = self.af(out)
        out = self.fc6(out)
        out = self.dropout(out)
        out = self.fc7(out)
        out = self.af(out)
        out = self.fc8(out)
        out = self.af(out)
        out = self.fc9(out)
        out = self.dropout(out)
        out = self.fc10(out)
        out = self.af(out)
        out = self.fc11(out)
        out = self.af(out)
        out = self.fc12(out)
        out = self.dropout(out)
        out = self.fc13(out)
        out = self.af(out)
        out = self.fc14(out)
        out = self.af(out)
        out = self.fc15(out)
        out = self.dropout(out)
        out = self.fc16(out)
        out = self.af(out)
        out = self.fc17(out)
        out = self.af(out)
        out = self.fc18(out)
        out = self.dropout(out)
        out = self.fc19(out)
        out = self.af(out)
        out = self.fc20(out)
        out = self.af(out)
        out = self.fc21(out)
        out = self.fc22(out)
        out = self.fc23(out)

        out=self.sigmoid(out)
        return(out)
```

**Weights:**

The model's weights are pre-trained and loaded into the neural network. Ensure that the weights file (`Heart_disease.CookieNeko`) is present in the application directory for accurate predictions.

### Data Collection

The app collects user responses to various health-related questions. These responses are converted into a numerical format suitable for input into the neural network.

### Risk Calculation

The risk percentage is computed by passing the processed input data through the neural network. The output of the neural network is a probability value between 0 and 1, which is then scaled to a percentage representing the likelihood of heart disease.

### UI/UX Design

The user interface is developed using Kivy, which provides a smooth and interactive experience. The design includes:

- **Questionnaire:** Sequential presentation of questions.
- **Progress Indicator:** Shows the user's progress through the questionnaire.
- **Result Display:** Shows the calculated risk percentage and provides additional information based on the result.

### Error Handling

Heartly includes error handling mechanisms to manage issues such as invalid inputs or failed model loading. Users are provided with clear instructions and prompts to correct any errors.

## Testing

Heartly has undergone rigorous testing to ensure functionality and accuracy. This includes:

- **Unit Tests:** Verifying the correctness of the neural network's prediction function.
- **UI Tests:** Ensuring that all user interface elements operate correctly across different platforms.
- **Integration Tests:** Confirming that data flows correctly from user input to risk calculation.

## Acknowledgements

Heartly was developed with the aim of making heart disease risk assessment accessible to everyone. Thanks to the Kivy and PyTorch communities for providing the tools and support that made this project possible.
