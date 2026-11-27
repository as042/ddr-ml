# ddr-ml

A small machine learning project with the goal to create a model that predicts the difficulty rating of a Dance Dance Revolution chart from its step-pattern statistics. This branch is my second attempt at this and is highly WIP. To see the previous attempt, view the `Legacy` branch.

## Building

The rating model is gradient-boosted trees from [LightGBM](https://github.com/microsoft/LightGBM), which is C++ and is compiled as part of `cargo build`. That takes CMake, a C++ compiler, and libclang with its headers. On Ubuntu:

```
sudo apt install cmake g++ libclang-dev
```
