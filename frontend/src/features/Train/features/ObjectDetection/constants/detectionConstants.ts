
import { grayscale } from "react-syntax-highlighter/dist/esm/styles/hljs";
import DetectionImageStep from "../components/DetectionImageStep";
import DetectionParametersStep from "../components/DetectionParametersStep";
import DetectionReviewStep from "../components/DetectionReviewStep";
import TabularTrainspace from "../components/DetectionTrainspace";
import React from "react";

export const TRAINSPACE_SETTINGS = {
  name: "Detection",
  steps: ["IMAGE", "PARAMETERS", "REVIEW"],
  component: TabularTrainspace,
} as const;

export const STEP_SETTINGS = {
  IMAGE: {
    name: "Image",
    optional: false,
    component: DetectionImageStep,
  },
  PARAMETERS: {
    name: "Parameters",
    optional: false,
    component: DetectionParametersStep,
    detectionTypes: [
      { label: "Rekognition", value: "rekognition" },
      { label: "YOLO", value: "yolo" },
    ],
    detectionProblemTypes : [
      { label: "Labels", value: "labels" },
      { label: "Celebrities", value: "celebrities" },
    ],
    detectionTransformValues: ["RandomHorizontalFlip", "ToTensor", "RandomVerticalFlip", "Resize", "GaussianBlur", "Grayscale", "Normalize", "AdjustContrast", "AdjustBrightness", "Affine"],
    detectionTransforms: {
      RandomHorizontalFlip: {
        display_name: "Random Horizontal Flip",
        object_name: "transforms.RandomHorizontalFlip",
        parameters: [
          {
            label: "probability",
            index: 0,
            parameter_name: "prob",
            min: 0,
            max: 1,
            type: "number",
          },
        ],
        label: "Random Horizontal Flip",
        value: "RandomHorizontalFlip",
      },
    
      ToTensor: {
        display_name: "To Tensor",
        object_name: "transforms.ToTensor",
        parameters: [],
        label: "To Tensor",
        value: "ToTensor",
      },

      RandomVerticalFlip: {
        display_name: "Random Vertical Flip",
        object_name: "transforms.RandomVerticalFlip",
        parameters: [
          {
            label: "probability",
            index: 0,
            parameter_name: "prob",
            min: 0,
            max: 1,
            type: "number",
          },
        ],
        label: "Random Vertical Flip",
        value: "RandomVerticalFlip",
      },
      Resize: {
        display_name: "Resize",
        object_name: "transforms.Resize",
        parameters: [
          {
            label: "size",
            index: 0,
            parameter_name: "(H, W)",
            min: 1,
            max: 1000,
            default: "(32, 32)",
            type: "tuple",
          },
        ],
        label: "Resize",
        value: "Resize",
      },
      GaussianBlur: {
        display_name: "Gaussian Blur",
        object_name: "transforms.GaussianBlur",
        parameters: [
          {
            label: "kernel size",
            index: 0,
            parameter_name: "kernel size",
            min: 1,
            max: 1000,
            type: "number",
          },
        ],
        label: "Gaussian Blur",
        value: "GaussianBlur",
      },
      Grayscale: {
        display_name: "Grayscale",
        object_name: "transforms.Grayscale",
        parameters: [],
        label: "Grayscale",
        value: "Grayscale",
      },
      Normalize: {
        display_name: "Normalize",
        object_name: "transforms.Normalize",
        parameters: [
          {
            label: "mean",
            index: 0,
            parameter_name: "mean",
            min: -1000,
            max: 1000,
            default: 0,
            type: "number",
          },
          {
            label: "std",
            index: 1,
            parameter_name: "std",
            min: -1000,
            max: 1000,
            default: 1,
            type: "number",
          },
        ],
        label: "Normalize",
        value: "Normalize",
      },
      AdjustContrast: {
        display_name: "AdjustContrast",
        object_name: "transforms.functional.adjust_contrast",
        transform_type: "functional",
        parameters: [
          {
            label: "contrast_factor",
            index: 0,
            parameter_name: "contrast_factor",
            min: 0,
            max: 10000,
            default: 0,
            type: "number",
          },
        ],
        label: "AdjustContrast",
        value: "AdjustContrast",
      },
      AdjustBrightness: {
        display_name: "AdjustBrightness",
        object_name: "transforms.functional.adjust_brightness",
        transform_type: "functional",
        parameters: [
          {
            label: "brightness_factor",
            index: 0,
            parameter_name: "brightness_factor",
            min: 0,
            max: 10000,
            default: 0,
            type: "number",
          },
        ],
        label: "AdjustBrightness",
        value: "AdjustBrightness",
      },
      Affine: {
        display_name: "Affine",
        object_name: "transforms.functional.affine",
        transform_type: "functional",
        parameters: [
          {
            label: "angle",
            index: 0,
            parameter_name: "angle",
            min: -180,
            max: 180,
            default: 0,
            type: "number",
          },
          {
            label: "translate",
            index: 1,
            parameter_name: "translate",
            min: -1000,
            max: 1000,
            default: "(0, 0)",
            type: "tuple",
          },
          {
            label: "scale",
            index: 2,
            parameter_name: "scale",
            min: 0,
            max: 1000,
            default: 1,
            type: "number",
          },
          {
            label: "shear",
            index: 3,
            parameter_name: "shear",
            min: -180,
            max: 180,
            default: "(0, 0)",
            type: "tuple",
          },
        ],
        label: "Affine",
        value: "Affine",
      },
    },
  },
  REVIEW: {
    name: "Review",
    optional: false,
    component: DetectionReviewStep,
  },
} as const;
