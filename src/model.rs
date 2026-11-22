use burn::{
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model {
    linears: Vec<Linear>,
    activation: Relu,
    dropout: Dropout,
}

/// A fully connected network, as the width of every layer from the inputs to the outputs. Each
/// hidden layer is followed by a ReLU, then dropout while training.
///
/// These are all 29 → 32 → 16 → 1:
/// - `ModelConfig::new(vec![29, 32, 16, 1])`
/// - `"29,32,16,1".parse()`
/// - `ModelConfig::funnel(29, 2, 32, 1)`
/// - `ModelConfig::stepped(29, 2, 32, 16, 1)`
#[derive(Config, Debug)]
pub struct ModelConfig {
    layer_sizes: Vec<usize>,
    /// Fraction of hidden activations zeroed each training step; none in validation mode
    #[config(default = 0.0)]
    pub dropout: f64,
}

impl ModelConfig {
    /// Hidden layers halve in width from `first_hidden_size`, rounding down.
    ///
    /// # Panics
    ///
    /// If halving would make a hidden layer 0 wide.
    pub fn funnel(num_inputs: usize, num_hidden_layers: usize, first_hidden_size: usize, num_outputs: usize) -> Self {
        let hidden = (0..num_hidden_layers).map(|i| first_hidden_size.checked_shr(i as u32).unwrap_or(0));
        let config = Self::new(std::iter::once(num_inputs).chain(hidden).chain([num_outputs]).collect());
        assert!(
            config.layer_sizes.iter().all(|&size| size > 0),
            "a funnel of {num_hidden_layers} hidden layers from {first_hidden_size} wide halves down to nothing"
        );
        config
    }

    /// Hidden layer widths step evenly from `first_hidden_size` to `last_hidden_size`, rounded.
    /// A single hidden layer is `first_hidden_size` wide.
    #[allow(dead_code)] // Only tests use it
    pub fn stepped(
        num_inputs: usize,
        num_hidden_layers: usize,
        first_hidden_size: usize,
        last_hidden_size: usize,
        num_outputs: usize,
    ) -> Self {
        let hidden = (0..num_hidden_layers).map(|i| {
            let fraction = if num_hidden_layers > 1 { i as f64 / (num_hidden_layers - 1) as f64 } else { 0.0 };
            let (first, last) = (first_hidden_size as f64, last_hidden_size as f64);
            (first + (last - first) * fraction).round() as usize
        });
        Self::new(std::iter::once(num_inputs).chain(hidden).chain([num_outputs]).collect())
    }

    #[allow(dead_code)] // Only tests use it
    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
    }

    /// Like `29 → 32 → 16 → 1`, which `parse` reads back.
    pub fn shape(&self) -> String {
        self.layer_sizes.iter().map(usize::to_string).collect::<Vec<_>>().join(" → ")
    }

    /// Returns the initialized model.
    ///
    /// # Panics
    ///
    /// If the config has fewer than 2 widths, a width of 0, or `dropout` outside `[0, 1)`.
    pub fn init(&self, device: &Device) -> Model {
        assert!(
            self.layer_sizes.len() >= 2 && self.layer_sizes.iter().all(|&size| size > 0),
            "a model needs at least an input and an output width, none of them 0, but got {:?}",
            self.layer_sizes
        );
        let linears = self.layer_sizes.windows(2).map(|pair| LinearConfig::new(pair[0], pair[1]).init(device)).collect();
        Model {
            linears,
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl std::str::FromStr for ModelConfig {
    type Err = String;

    /// Layer widths separated by `,`, `->` or `→`.
    fn from_str(text: &str) -> Result<Self, String> {
        let sizes = text
            .replace("->", ",")
            .replace('→', ",")
            .split(',')
            .map(|size| match size.trim().parse::<usize>() {
                Ok(size) if size > 0 => Ok(size),
                _ => Err(format!("{:?} isn't a layer width (in {text:?})", size.trim())),
            })
            .collect::<Result<Vec<usize>, String>>()?;
        if sizes.len() < 2 {
            return Err(format!("a model needs at least an input and an output width, but {text:?} has {}", sizes.len()));
        }
        Ok(Self::new(sizes))
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
            x = self.dropout.forward(x);
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
        // (config, every layer's width)
        let cases = [
            (ModelConfig::new(vec![29, 40, 8, 16, 1]), vec![29, 40, 8, 16, 1]),
            (ModelConfig::funnel(29, 2, 32, 1), vec![29, 32, 16, 1]),
            (ModelConfig::funnel(10, 3, 48, 2), vec![10, 48, 24, 12, 2]),
            (ModelConfig::funnel(5, 4, 20, 1), vec![5, 20, 10, 5, 2, 1]),
            (ModelConfig::funnel(5, 0, 32, 1), vec![5, 1]),
            (ModelConfig::stepped(29, 2, 32, 16, 1), vec![29, 32, 16, 1]),
            (ModelConfig::stepped(42, 10, 100, 10, 5), vec![42, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]),
            (ModelConfig::stepped(3, 3, 10, 5, 1), vec![3, 10, 8, 5, 1]),
            (ModelConfig::stepped(3, 1, 8, 4, 1), vec![3, 8, 1]),
            (ModelConfig::stepped(3, 0, 8, 4, 1), vec![3, 1]),
        ];
        for (config, sizes) in cases {
            assert_eq!(config.layer_sizes(), sizes, "{config:?}");
            let model = config.init(&device);
            let built: Vec<[usize; 2]> = model.linears.iter().map(|linear| linear.weight.dims()).collect();
            let expected: Vec<[usize; 2]> = sizes.windows(2).map(|pair| [pair[0], pair[1]]).collect();
            assert_eq!(built, expected, "{config:?}");
        }
    }

    #[test]
    fn parses_layer_widths() {
        for text in ["29,32,16,1", " 29, 32 ,16,1 ", "29->32->16->1", "29 -> 32 -> 16 -> 1", "29 → 32 → 16 → 1"] {
            let config: ModelConfig = text.parse().unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert_eq!(config.layer_sizes(), [29, 32, 16, 1], "{text:?}");
        }
        // `shape` writes what `parse` reads
        let funnel = ModelConfig::funnel(29, 3, 64, 1);
        assert_eq!(funnel.shape(), "29 → 64 → 32 → 16 → 1");
        assert_eq!(funnel.shape().parse::<ModelConfig>().unwrap().layer_sizes(), funnel.layer_sizes());

        for text in ["", "29", "29,,1", "29,a,1", "29,0,1", "29,-3,1", "29, 32.5, 1"] {
            assert!(text.parse::<ModelConfig>().is_err(), "{text:?} should fail to parse");
        }
    }

    #[test]
    #[should_panic(expected = "halves down to nothing")]
    fn funnel_rejects_halving_to_nothing() {
        ModelConfig::funnel(3, 4, 4, 1); // 4, 2, 1, then 0
    }

    #[test]
    fn dropout_only_while_training() {
        let device = DevicePreference::default().device();
        device.seed(1);
        let x = Tensor::<2>::ones([8, 3], &device);
        let outputs = |model: &Model| model.forward(x.clone()).into_data().try_to_vec::<f32>().unwrap();

        let model = ModelConfig::new(vec![3, 64, 1]).with_dropout(0.5).init(&device);
        assert_ne!(outputs(&model), outputs(&model), "dropout should differ between training steps");
        let valid = model.valid();
        let first = outputs(&valid);
        assert_eq!(first, outputs(&valid));
        assert!(first.iter().all(|&y| y == first[0]), "identical rows should give identical outputs");

        let no_dropout = ModelConfig::new(vec![3, 64, 1]).init(&device);
        assert_eq!(outputs(&no_dropout), outputs(&no_dropout));
    }

    #[test]
    fn test_model_forward() {
        let device = DevicePreference::default().device();
        let model = ModelConfig::stepped(3, 9, 100, 10, 5).init(&device);
        let output = model.forward(Tensor::<1>::from_data([0.5, -1.2, 100.0], &device));

        assert_eq!(output.dims()[0], 5);
    }
}