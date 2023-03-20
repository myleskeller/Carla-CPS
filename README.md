# Carla-CPS

This is a client-side python file that interacts with the PythonAPI component of carla and attempts to simulate data propagation across actors within a map in the form of vehicles, walkers, and specific semantic nodes. 

This is accomplished entirely on the client-side by exploiting the functionality of the built-in `AObstacleDetectionSensor` (_because attempting to add a custom sensor to the server directly and compiling manually using the latest build appears to be completely broken_).

To run it:
 - start your carla server instance (you can download precompiled binaries off of their official github)
 - open a command prompt
 - `cd` to the folder containing `manual_control_AObstacleDetectionSensor.py`
 - enter `python manual_control_AObstacleDetectionSensor.py --sync`
