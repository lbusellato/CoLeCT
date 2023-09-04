# CoLeCT - Control and Learning of Contact Transitions

This repository contains the code for the CoLeCT project. The project is an end-to-end learning-from-demonstration framework that covers:

* Recording demonstrations of trajectories defined by poses and force profiles.
* Post-processing the recorded demonstrations using Soft-DTW to temporally align them.
* Probabilistic modeling of the demonstration database using Gaussian Mixture Models (GMMs) and Gaussian Mixture Regression (GMR).
* Learning the probabilistic features of the trajectories using Kernelized Movement Primitives (KMP).
* Reproduction on an UR5e robotic manipulator.

## Table of contents

- [CoLeCT - Control and Learning of Contact Transitions](#colect---control-and-learning-of-contact-transitions)
  * [Setup](#setup)
  * [Demonstration recording](#demonstration-recording)
  * [Dataset postprocessing](#dataset-postprocessing)
  * [Learning from demonstration](#learning-from-demonstration)
  * [Reproduction](#reproduction)

## Setup

The package requirements for the project are listed in [requirements.txt](https://github.com/lbusellato/colect/blob/main/requirements.txt). It is highly suggested to use a Python3 virtual environment, which can be set up as follows:

    python3 -m venv colect-venv
    source colect-venv/bin/activate
    python3 -m pip install -r requirements.txt
    source colect-venv/bin/activate

## Demonstration recording

The demonstration recording part of the framework was built around a custom-made handheld recording tool. 

<p align="center">
  <img src="https://github.com/lbusellato/colect/blob/main/media/render.png" />
</p>

## Dataset postprocessing

## Learning from demonstration

## Reproduction