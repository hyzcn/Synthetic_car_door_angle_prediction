# Synthetic_car_door_angle_prediction
This experiment provides an application of deformable synthetic shapenet cars.

By doing the car door angle prediction, we make use of the intermediate data provided by synthetic data and find out that intermediate supervision contributes to improving model's generalization ability.
## Goal
* Predict synthetic car doors' open angle
* Show effectiveness of different types of supervision
* Transfer between synthetic and real car images
## Data
* CAD car models from UE4 market
* Deformable shapenet cars
## Model
ResNet18 with FCN and multi-task architecture.

You can check [this slides](https://docs.google.com/presentation/d/1s9xKb6-U3IOtFZs1-ZPRSswHbBh03IWLhYbnwgss2y0/edit?usp=sharing) for more details.
