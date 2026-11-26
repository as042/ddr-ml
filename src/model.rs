use burn::{
    module::{Param, ParamId},
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model {
    linears: Vec<Linear>,
    activation: Relu,
    dropout: Dropout,
    /// See `ModelConfig::input_clip`
    input_clip: f64,
}

/// A fully connected network, as the width of every layer from the inputs to the outputs. Each
/// hidden layer is followed by a ReLU, then dropout while training. Inputs can be capped first.
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
    /// Inputs (scaled, as the model gets them) are capped at plus or minus this, in training
    /// and prediction alike, so a statistic far outside the training range can't swing a
    /// prediction without bound. 0 for no cap
    /// (A config saved before this existed has none: see `ModelConfig::read`.)
    #[config(default = 0.0)]
    pub input_clip: f64,
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

    /// A saved config. burn's `Config::load` needs every field, so one saved before `input_clip`
    /// existed gets it here, as 0.
    pub fn read(path: impl AsRef<std::path::Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut fields: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        if let Some(fields) = fields.as_object_mut() {
            fields.entry("input_clip").or_insert(0.0.into());
        }
        Ok(serde_json::from_value(fields)?)
    }

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
    /// If the config has fewer than 2 widths, a width of 0, `dropout` outside `[0, 1)`, or a
    /// negative `input_clip`.
    pub fn init(&self, device: &Device) -> Model {
        assert!(self.input_clip >= 0.0, "input_clip must be 0 (none) or more, not {}", self.input_clip);
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
            input_clip: self.input_clip,
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

    /// A model that gives `x` what this one gives `x · weights + offsets`, by folding that map
    /// into the first layer. `weights` is `[new_num_inputs, num_inputs]`, `offsets`
    /// `[1, num_inputs]`.
    ///
    /// # Panics
    ///
    /// If the model caps its inputs: the cap would then apply to the new inputs, not the old.
    pub fn with_input_map(mut self, weights: Tensor<2>, offsets: Tensor<2>) -> Self {
        assert_eq!(self.input_clip, 0.0, "a model that caps its inputs can't take an input map");
        let weight = self.linears[0].weight.val();
        let [_, width] = weight.dims();
        let shift = offsets.matmul(weight.clone()).reshape([width]);
        let bias = match &self.linears[0].bias {
            Some(bias) => bias.val() + shift,
            None => shift,
        };
        let mut first = LinearConfig::new(weights.dims()[0], width).init(&weight.device());
        first.weight = Param::initialized(ParamId::new(), weights.matmul(weight));
        first.bias = Some(Param::initialized(ParamId::new(), bias));
        self.linears[0] = first;
        self
    }

    pub fn forward<const D: usize>(&self, mut x: Tensor<D>) -> Tensor<D> {
        if self.input_clip > 0.0 {
            x = x.clamp(-self.input_clip, self.input_clip);
        }
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
    fn caps_inputs_when_asked() {
        let device = DevicePreference::default().device();
        let inputs = |values: [f32; 2]| Tensor::<2>::from_data(TensorData::new(values.to_vec(), [1, 2]), &device);
        let output = |model: &Model, values| model.forward(inputs(values)).into_data().try_to_vec::<f32>().unwrap();
        let config = ModelConfig::new(vec![2, 3, 1]);
        assert_eq!(config.input_clip, 0.0, "no cap by default");
        let capped = config.clone().with_input_clip(1.0).init(&device);
        assert_eq!(output(&capped, [5.0, -7.0]), output(&capped, [1.0, -1.0]));
        assert_ne!(output(&capped, [0.5, -7.0]), output(&capped, [1.0, -1.0]));
        let uncapped = config.init(&device);
        assert_ne!(output(&uncapped, [5.0, -7.0]), output(&uncapped, [1.0, -1.0]));
        // Configs saved before the cap load without one
        let path = std::env::temp_dir().join(format!("ddr-trainer-old-config-{}.json", std::process::id()));
        std::fs::write(&path, r#"{"layer_sizes": [335, 2048, 1], "dropout": 0.2}"#).unwrap();
        let old = ModelConfig::read(&path);
        let capped = ModelConfig::new(vec![3, 1]).with_input_clip(5.0);
        capped.save(&path).unwrap();
        let saved = ModelConfig::read(&path);
        std::fs::remove_file(&path).unwrap();
        let old = old.unwrap();
        assert_eq!((old.layer_sizes(), old.dropout, old.input_clip), (&[335, 2048, 1][..], 0.2, 0.0));
        assert_eq!(saved.unwrap().input_clip, 5.0);
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
    fn folds_an_input_map_into_the_first_layer() {
        let device = DevicePreference::default().device();
        let model = ModelConfig::new(vec![3, 5, 2]).init(&device).valid();
        let device = model.linears[0].weight.val().device();
        let weights = Tensor::<2>::from_data([[1.0, 0.5, -2.0], [0.0, 3.0, 1.0], [-1.0, 1.0, 0.25], [2.0, -0.5, 0.0]], &device);
        let offsets = Tensor::<2>::from_data([[0.5, -1.0, 2.0]], &device);
        let x = Tensor::<2>::from_data([[0.3, -1.0, 2.0, 0.7], [5.0, 0.0, -3.0, 1.5]], &device);

        let expected = model.forward(x.clone().matmul(weights.clone()) + offsets.clone());
        let mapped = model.with_input_map(weights, offsets);
        assert_eq!(mapped.num_inputs(), 4);
        assert_eq!(mapped.num_outputs(), 2);
        let max_diff = (mapped.forward(x) - expected).abs().max().into_scalar::<f32>();
        assert!(max_diff < 1e-5, "outputs differ by {max_diff}");
    }

    #[test]
    fn test_model_forward() {
        let device = DevicePreference::default().device();
        let model = ModelConfig::stepped(3, 9, 100, 10, 5).init(&device);
        let output = model.forward(Tensor::<1>::from_data([0.5, -1.2, 100.0], &device));

        assert_eq!(output.dims()[0], 5);
    }
}