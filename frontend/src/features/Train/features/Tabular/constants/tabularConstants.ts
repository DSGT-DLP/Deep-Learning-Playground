
import * as React from 'react';
import { styled } from '@mui/material/styles';
import Button from '@mui/material/Button';
import Tooltip, { TooltipProps, tooltipClasses } from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import TabularDatasetStep from "../components/TabularDatasetStep";
import TabularParametersStep from "../components/TabularParametersStep";
import TabularReviewStep from "../components/TabularReviewStep";
import TabularTrainspace from "../components/TabularTrainspace";

export const TRAINSPACE_SETTINGS = {
  name: "Tabular",
  steps: ["DATASET", "PARAMETERS", "REVIEW"],
  component: TabularTrainspace,
} as const;

export const STEP_SETTINGS = {
  DATASET: {
    name: "Dataset",
    optional: false,
    component: TabularDatasetStep,
    defaultDatasets: [
      { label: "Iris", value: "IRIS" },
      { label: "California Housing", value: "CALIFORNIAHOUSING" },
      { label: "Diabetes", value: "DIABETES" },
      { label: "Digits", value: "DIGITS" },
      { label: "Wine", value: "WINE" },
    ],
  },
  PARAMETERS: {
    name: "Parameters",
    optional: false,
    component: TabularParametersStep,
    problemTypes: [
      { label: "Classification", value: "CLASSIFICATION" },
      { label: "Regression", value: "REGRESSION" },
    ],
    criterions: [
      {
        label: "L1 (Absolute Error) Loss",
        value: "L1LOSS",
        objectName: "nn.L1Loss()",
        problemType: "REGRESSION",
      },
      {
        label: "Mean Squared Error Loss",
        value: "MSELOSS",
        objectName: "nn.MSELoss()",
        problemType: "REGRESSION",
      },
      {
        label: "Binary Cross-Entropy Loss",
        value: "BCELOSS",
        objectName: "nn.BCELoss()",
        problemType: "CLASSIFICATION",
      },
      {
        label: "Cross-Entropy Loss",
        value: "CELOSS",
        objectName: "nn.CrossEntropyLoss(reduction='mean')",
        problemType: "CLASSIFICATION",
      },
    ],
    optimizers: [
      { label: "Stochastic Gradient Descent", value: "SGD" },
      { label: "Adam Optimization", value: "Adam" },
    ],
    layerValues: ["LINEAR", "RELU", "TANH", "SOFTMAX", "SIGMOID", "LOGSOFTMAX"],

    layers: {
      LINEAR: {
        label: "Linear",
        objectName: "nn.Linear",
        parameters: [
          {
            label: "Input Size",
            min: 1,
            max: 1600,
            required: true,
            type: "number",
          },
          {
            label: "Output Size",
            min: 1,
            max: 1600,
            required: true,
            type: "number",
          },
        ],
        description: "A linear layer performs a mathematical operation called linear transformation on a set of input values. It applies a combination of scaling and shifting to the input values, resulting in a new set of transformed values as output."
      },
      RELU: {
        label: "ReLU",
        objectName: "nn.ReLU",
        parameters: [],
        description: "ReLU, short for Rectified Linear Unit, is an activation function that acts like a filter that selectively allows positive numbers to pass through unchanged, while converting negative numbers to zero."
      },
      TANH: {
        label: "Tanh",
        objectName: "nn.Tanh",
        parameters: [],
        description: "The tanh function maps input numbers to a range between -1 and 1, emphasizing values close to zero while diminishing the impact of extremely large or small numbers, making it useful for capturing complex patterns in data."
      },
      SOFTMAX: {
        label: "Softmax",
        objectName: "nn.Softmax",
        parameters: [
          {
            label: "Dimension",
            min: -3,
            max: 2,
            required: true,
            type: "number",
          },
        ],
        description: "The softmax function takes a set of numbers as input and converts them into a probability distribution, assigning higher probabilities to larger numbers and lower probabilities to smaller numbers, making it useful for multi-class classification tasks."
      },
      SIGMOID: {
        label: "Sigmoid",
        objectName: "nn.Sigmoid",
        parameters: [],
        description: "The sigmoid function takes any input number and squeezes it to a range between 0 and 1, effectively converting it into a probability-like value, often used for binary classification tasks and as an activation function in neural networks."
      },
      LOGSOFTMAX: {
        label: "LogSoftmax",
        objectName: "nn.LogSoftmax",
        parameters: [
          {
            label: "Dimension",
            min: -3,
            max: 2,
            required: true,
            type: "number",
          },
        ],
        description: "The logsoftmax function converts a set of numbers into a probability distribution using the softmax function, and then applies a logarithm to the resulting probabilities. It is commonly used for multi-class classification tasks as an activation function and to calculate the logarithmic loss during neural network training."
      }
    },
  },
  REVIEW: {
    name: "Review",
    optional: false,
    component: TabularReviewStep,
  },
} as const;
