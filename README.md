# CoLeCT - Control and Learning of Contact Transitions

This repository contains the code for the CoLeCT project. The project is an end-to-end learning-from-demonstration framework that covers:

* Recording demonstrations of trajectories defined by poses and force profiles.
* Post-processing the recorded demonstrations using Soft-DTW to temporally align them.
* Probabilistic modeling of the demonstration database using Gaussian Mixture Models (GMMs) and Gaussian Mixture Regression (GMR).
* Learning the probabilistic features of the trajectories using Kernelized Movement Primitives (KMP).
* Uncertainty-aware controller gain computation using a Linear-Quadratic Regulator (LQR).
* Reproduction on an UR5e robotic manipulator with a parallel position/force controller.

## Table of contents

- [CoLeCT - Control and Learning of Contact Transitions](#colect---control-and-learning-of-contact-transitions)
  * [Setup](#setup)
  * [TODO](#TODO)
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

To use the demonstration recording pipeline, an OptiTrack camera system with Motive version 2.x is needed.

## TODO

* #BUG KMP w/ pose as input, force as output is not working right
* #FEATURE KMP w/ null-space projector
* #FEATURE Hyperparameter tuning for GMM, GMR and KMP
* #FEATURE Threaded LQR gain computation
* #ENHANCEMENT Recording both robot base and handheld tool at the same time, making the coordinate shift part of the recording instead of the postprocessing
* #ENHANCEMENT Motive server IP autodiscovery

## Demonstration recording

The demonstration recording part of the framework was built around a custom-made handheld recording tool. 

<p align="center">
  <img src="media/render.png" width="600" height="400" >
</p>

The tool has six OptiTrack markers used for recording the position and orientation, and a Wittenstein HEX12 6DOF force/torque sensor mounted on the end-effector.

Demonstration recording is handled by the Recorder class. A sample usage can be found in [samples/demonstration_recording.py](https://github.com/lbusellato/colect/blob/main/requirements.txt). 

The pipeline for demonstration recording is as follows:

* Prepare the workspace:
  * Place at least three OptiTrack markers around the base of the robot. This is done to record the position of the base frame of the robot. 
  * Place the handheld tool into the workspace.
  * Connect the force sensor to the system running the code.
* In Motive:
  * Select all the markers on the robot base and create a new rigid body from them. 
  * Select all the markers of the handheld tool and create a new rigid body from them.
  * In the *Streaming* tab, set the *Local Interface* to the machine's external IP and set *Transmission Type* to *Unicast*.
  * In the *Assets* tab, deselect all rigid bodies except the one you want to record.
* Demonstration recording:
  * Create an instance of the Recorder class, supplying as argument the same IP address you set in Motive.
  * Start the execution by calling Recorder.run().
  * Press 'r' + ENTER to start the recording. Press 's' + ENTER to stop the recording. Press 'q' + ENTER to quit.

Besides the trajectory recording with the handheld tool, a recording must be made of the rigid body placed on the robot base, so that the trajectory coordinates can be converted to the robot base frame.

Recordings are saved in .csv format in the recordings folder. In principle, the recording step can be done with other methods, as long as the produced .csv file has the header:

    timestamp,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,force_x,force_y,force_z,torque_x,torque_y,torque_z
  

## Dataset postprocessing

## Learning from demonstration

## Reproduction