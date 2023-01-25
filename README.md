# Heading Project Code
This code was developed in conjunction with this [paper](https://www.biorxiv.org/content/10.1101/2022.03.25.485865v1).
However, the code was written to create an easily expandable toolkit for processing two-photon calcium imaging data taken from a mouse on the Neurotar Mobile Home Cage system (MHC), allowing for flexible module-based analyses of a wide array of experimental set ups on the MHC.

This code was written by Kevin Sit.

## Installation
Clone this repository containing a bunch of small helper functions first: [GitHub Link](https://github.com/kevinksit/GeneralHelperCode).
Then clone this repository and make sure both are on your MATLAB path.

## Code Structure
### Main Code
The code is based around a set of abstract classes which provide modular functionality into each "Experiment".
Because the classes are abstract, they cannot be used indepednently, and must be combined together to work. There are two main types of abstract classes:

- **Cues**: These classes serve to define the types of cues that are present in the MHC. For example, this may represent a single white cue card, or paired symmetric cues.
- **Controls**: These classes describe the movement of the cage. In some experiments, the cage may be fixed and only allowed to rotate, in others the cage may be allowed to translate without rotation, and others still may have fully free movement.

Each type of experiment is described in an "Experiment" class which inherits **one** "Cue" and **one** "Control" class.
Together, these abstract classes describe the experimental design and allow for further expansion of experiment-specific analyses.

The following flowchart provides a graphical description of each of the classes and their respective methods:

![class flowchart](./flowchart.png)

**Cue** and **Control** are the parent abstract classes that contain many of the key methods.
**SingleCue** and **DualCue** are childen of **Cue** which either alter or expand the methods present in **Cue**.
**ControlledRotation** and **FreelyMoving** do the same for **Control**.
All these classes are abstract classes, and therefore are not meant to be instantiated on their own.

Instead, the "real classes" are made of a combination of one **Cue** and one **Control**.
Three examples are shown here, related to analyses performed in the associated paper. Each "Experiment" takes a different combination of **Cue** and **Control** to properly alter analyses to fit the experimental design.

### Description of methods in each class
- **Cues**
	- *getTuningCurves*: Returns the tuning curve for each neuron, defined as the averaged activity for each heading bin.
	- *calculatePreferredDirection*: Returns the preferred heading (in radians) of each neuron.
Has several methods for calculating each neurons preferred heading, but 'fit' is recommended.
	- *calculateFlipScore*: Returns the flip score for each neuron (see [Jacob et. al. 2017](https://pubmed.ncbi.nlm.nih.gov/27991898/)).
	- *calculateTuningFits*: Fits a one or two term Gaussian (depends on the number of cues) to each neuron.
- **Control**
	- *getNeuralData*: Returns the timeseries of neural data for each neuron (either DFF or spikes)
	- *getHeading*: Returns the recorded chamber orientation (heading) for the recording.
	- *calculateHeadDirection*: Determines whether or not a neuron is heading selective or not.

There are other methods present in each class, generally as helper or "checker" functions. All the main analyses are described above.

### Usage
Please see `example_usage.mat` for a guided example of how to use this toolbox.

## Expanding and developing new experiments
### Creating new Experiments
Creating a new type of experiment is easy.
Simply create your "Experiment" subclass, and inherit from one Cue and one Control class to combine the functionality to create an "Experiment".

### Creating new Cues
Keep in mind the necessary methods and properties required in the Cue class, and create a new Cue subclass that inherits from the main Cue class.
See SingleCue and DualCue for examples of expanded functionality on top of the base Cue class.

### Creating new Controls
Controls are relatively fixed, but can be added to in case there's other highly specific control schemes.
Like with the Cues, make sure to inherit from the Control parent class and expand functionality as desired.

## Dependencies
- Image Processing Toolbox
- Signal Processing Toolbox 
