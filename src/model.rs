use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model {
    linears: Vec<Linear>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_inputs: usize,
    num_hidden_layers: usize,
    first_hidden_size: usize,
    last_hidden_size: usize,
    num_outputs: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init(&self, device: &Device) -> Model {
        fn layer_size(hidden_idx: usize, num_hidden_layers: usize, first_hidden_size: usize, last_hidden_size: usize) -> usize {
            (first_hidden_size as f64 + (last_hidden_size as f64 - first_hidden_size as f64) * (hidden_idx as f64 / num_hidden_layers as f64)).round() as usize
        }

        let mut next_size = layer_size(0, self.num_hidden_layers, self.first_hidden_size, self.last_hidden_size);
        let mut linears = vec![LinearConfig::new(self.num_inputs, next_size)];
        for l in 0..self.num_hidden_layers {
            let old_size = next_size;
            next_size = layer_size(l + 1, self.num_hidden_layers, self.first_hidden_size, self.last_hidden_size);
            let linear = LinearConfig::new(old_size, next_size);
            linears.push(linear);
        }

        linears.push(LinearConfig::new(next_size, self.num_outputs));

        // println!("Linear config list: {:?}", linears);

        let linears = linears.iter().map(|x| x.init(&device)).collect::<Vec<Linear>>();

        Model {
            linears,
            activation: Relu::new(),
        }
    }
}

impl Model {
    /// Number of input columns the model expects.
    pub fn num_inputs(&self) -> usize {
        self.linears[0].weight.dims()[0]
    }

    /// Number of output columns the model produces.
    pub fn num_outputs(&self) -> usize {
        self.linears.last().unwrap().weight.dims()[1]
    }

    pub fn forward<const D: usize>(&self, mut x: Tensor<D>) -> Tensor<D> {
        for l in 0..(self.linears.len() - 1) {
            x = self.linears[l].forward(x);
            x = self.activation.forward(x);
        }

        self.linears.last().unwrap().forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DevicePreference;

    #[test]
    fn test_model_config() {
        let device = DevicePreference::default().device();
        let model = ModelConfig::new(42, 9, 100, 10, 5).init(&device);
        let expected_model = Model {
            linears: vec![
                LinearConfig::new(42, 100).init(&device),
                LinearConfig::new(100, 90).init(&device),
                LinearConfig::new(90, 80).init(&device),
                LinearConfig::new(80, 70).init(&device),
                LinearConfig::new(70, 60).init(&device),
                LinearConfig::new(60, 50).init(&device),
                LinearConfig::new(50, 40).init(&device),
                LinearConfig::new(40, 30).init(&device),
                LinearConfig::new(30, 20).init(&device),
                LinearConfig::new(20, 10).init(&device),
                LinearConfig::new(10, 5).init(&device),
            ],
            activation: Relu::new(),
        };

        assert_eq!(model.linears.len(), expected_model.linears.len());
        assert_eq!(format!("{:?}", model.activation), format!("{:?}", expected_model.activation));
    }

    #[test]
    fn test_model_forward() {
        let device = DevicePreference::default().device();
        let model = ModelConfig::new(3, 9, 100, 10, 5).init(&device);
        let output = model.forward(Tensor::<1>::from_data([0.5, -1.2, 100.0], &device));

        assert_eq!(output.dims()[0], 5);
    }
}