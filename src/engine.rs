use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{Array, ArrayD};
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::Tensor as RtenTensor;

use crate::types::{DetConfig, EngineConfig, RecConfig};

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("RTen load error: {0}")]
    Load(#[from] rten::LoadError),

    #[error("RTen run error: {0}")]
    Run(#[from] rten::RunError),

    #[error("Engine error: {0}")]
    Engine(String),

    #[cfg(feature = "use-opencv")]
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),

    #[error("Image processing error: {0}")]
    ImageError(String),

    #[error("Invalid input shape")]
    InvalidInputShape,

    #[error("Preprocess error: {0}")]
    Preprocess(String),

    #[error("Shape error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),

    #[error("Output error: {0}")]
    OutputError(String),
}

impl From<Box<dyn std::error::Error>> for EngineError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        EngineError::ImageError(err.to_string())
    }
}

/// Abstract inference session interface allowing alternative backends
/// (e.g. RTen, ONNX Runtime, Tract, Candle, or custom hardware accelerators).
pub trait InferenceSession: Send {
    /// Execute inference with a single input tensor returning primary output tensor.
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError>;

    /// Execute inference with a single input tensor returning all named output tensors.
    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError>;

    /// Execute inference with named input tensors returning all named output tensors.
    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError>;

    /// Optional metadata lookup for embedded character list.
    fn get_character_list(&self, _key: &str) -> Option<Vec<String>> {
        None
    }

    /// Check if metadata key exists.
    fn have_key(&self, _key: &str) -> bool {
        false
    }
}

impl InferenceSession for Box<dyn InferenceSession> {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        (**self).run(input)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        (**self).run_all(input)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        (**self).run_with_inputs(inputs)
    }

    fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        (**self).get_character_list(key)
    }

    fn have_key(&self, key: &str) -> bool {
        (**self).have_key(key)
    }
}

/// Generic engine session enum supporting both default pure-Rust RTen
/// and any custom backend implementing `InferenceSession`.
pub enum EngineSession {
    Rten(Box<RtenSession>),
    Custom(Box<dyn InferenceSession>),
}

#[deprecated(note = "MnnSession is deprecated, please use EngineSession")]
pub type MnnSession = EngineSession;

impl EngineSession {
    pub fn from_path(model_path: &Path, engine_cfg: &EngineConfig) -> Result<Self, EngineError> {
        let session = RtenSession::from_path(model_path, engine_cfg)?;
        Ok(Self::Rten(Box::new(session)))
    }

    pub fn from_det_config(cfg: &DetConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_rec_config(cfg: &RecConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_cls_config(cfg: &crate::types::ClsConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_orient_config(cfg: &crate::types::OrientConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_custom(session: impl InferenceSession + 'static) -> Self {
        Self::Custom(Box::new(session))
    }

    pub fn from_boxed(session: Box<dyn InferenceSession>) -> Self {
        Self::Custom(session)
    }

    pub fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        match self {
            Self::Rten(s) => s.run(input),
            Self::Custom(s) => s.run(input),
        }
    }

    pub fn run_all(
        &mut self,
        input: ArrayD<f32>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        match self {
            Self::Rten(s) => s.run_all(input),
            Self::Custom(s) => s.run_all(input),
        }
    }

    pub fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        match self {
            Self::Rten(s) => s.run_with_inputs(inputs),
            Self::Custom(s) => s.run_with_inputs(inputs),
        }
    }

    pub fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        match self {
            Self::Rten(s) => s.get_character_list(key),
            Self::Custom(s) => s.get_character_list(key),
        }
    }

    pub fn have_key(&self, key: &str) -> bool {
        match self {
            Self::Rten(s) => s.have_key(key),
            Self::Custom(s) => s.have_key(key),
        }
    }

    /// Raw string value of a custom model-metadata key (ONNX `metadata_props`).
    /// `.rten` files do not keep custom metadata, and custom backends expose
    /// none, so this returns `None` for them.
    pub fn metadata_value(&self, key: &str) -> Option<String> {
        match self {
            Self::Rten(s) => s.metadata_value(key),
            Self::Custom(_) => None,
        }
    }
}

impl InferenceSession for EngineSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        self.run(input)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_all(input)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_with_inputs(inputs)
    }

    fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        self.get_character_list(key)
    }

    fn have_key(&self, key: &str) -> bool {
        self.have_key(key)
    }
}

pub struct RtenSession {
    model: Model,
}

fn resolve_model_path(path: &Path) -> PathBuf {
    if path.exists() {
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ext.eq_ignore_ascii_case("rten") || ext.eq_ignore_ascii_case("onnx") {
                return path.to_path_buf();
            }
        }
    }

    let rten_cand = path.with_extension("rten");
    if rten_cand.exists() {
        return rten_cand;
    }

    let onnx_cand = path.with_extension("onnx");
    if onnx_cand.exists() {
        return onnx_cand;
    }

    if path.exists() {
        return path.to_path_buf();
    }

    if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
        let is_det = file_name.starts_with("det");
        let is_rec = file_name.starts_with("rec");
        let base = if is_det {
            "det"
        } else if is_rec {
            "rec"
        } else {
            ""
        };

        if !base.is_empty() {
            for dir in &["models/PPOCR_v6_tiny", "models/PPOCR_v6", "models"] {
                let p_rten = PathBuf::from(format!("{}/{}.rten", dir, base));
                if p_rten.exists() {
                    return p_rten;
                }
                let p_onnx = PathBuf::from(format!("{}/{}.onnx", dir, base));
                if p_onnx.exists() {
                    return p_onnx;
                }
            }
        }
    }

    path.to_path_buf()
}

/// Load the model of a document-analysis stage (`stage` is used in error
/// messages, for example `"layout"`). The path is resolved like any other model
/// path; a missing file or an extension other than `.rten`/`.onnx` is reported
/// with the stage name instead of an opaque loader error.
pub(crate) fn load_stage_session(
    stage: &str,
    model_path: &Path,
    cfg: &EngineConfig,
) -> Result<EngineSession, EngineError> {
    let resolved = resolve_model_path(model_path);
    if !resolved.exists() {
        return Err(EngineError::Engine(format!(
            "{stage} model not found: {}",
            model_path.display()
        )));
    }
    let ok_ext = resolved
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("rten") || e.eq_ignore_ascii_case("onnx"));
    if !ok_ext {
        return Err(EngineError::Engine(format!(
            "{stage} model must be a .rten or .onnx file: {}",
            model_path.display()
        )));
    }
    EngineSession::from_path(&resolved, cfg)
}

fn convert_output_to_array(value: &rten::Value) -> Result<ArrayD<f32>, EngineError> {
    match value {
        rten::Value::FloatTensor(t) => {
            let shape = t.shape().to_vec();
            let data: Vec<f32> = t.iter().copied().collect();
            let array = Array::from_shape_vec(shape, data)?;
            Ok(array.into_dyn())
        }
        rten::Value::Int32Tensor(t) => {
            let shape = t.shape().to_vec();
            let data: Vec<f32> = t.iter().map(|&x| x as f32).collect();
            let array = Array::from_shape_vec(shape, data)?;
            Ok(array.into_dyn())
        }
        rten::Value::Int8Tensor(t) => {
            let shape = t.shape().to_vec();
            let data: Vec<f32> = t.iter().map(|&x| x as f32).collect();
            let array = Array::from_shape_vec(shape, data)?;
            Ok(array.into_dyn())
        }
        rten::Value::UInt8Tensor(t) => {
            let shape = t.shape().to_vec();
            let data: Vec<f32> = t.iter().map(|&x| x as f32).collect();
            let array = Array::from_shape_vec(shape, data)?;
            Ok(array.into_dyn())
        }
        _ => Err(EngineError::OutputError(
            "Unsupported output tensor type".into(),
        )),
    }
}

impl RtenSession {
    pub fn from_det_config(cfg: &DetConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_rec_config(cfg: &RecConfig) -> Result<Self, EngineError> {
        Self::from_path(&cfg.model_path, &cfg.engine_cfg)
    }

    pub fn from_path(model_path: &Path, _engine_cfg: &EngineConfig) -> Result<Self, EngineError> {
        let resolved = resolve_model_path(model_path);
        let model = Model::load_file(&resolved)?;
        Ok(Self { model })
    }

    pub fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        let shape: Vec<usize> = input.shape().to_vec();
        let data: Vec<f32> = if let Some(slice) = input.as_slice() {
            slice.to_vec()
        } else {
            input.iter().copied().collect()
        };

        let tensor = RtenTensor::<f32>::from_data(&shape, data);
        let in_ids = self.model.input_ids();
        if in_ids.is_empty() {
            return Err(EngineError::OutputError("Model has no inputs".into()));
        }
        let in_id = in_ids[0];

        let out_ids = self.model.output_ids();
        let outputs = self.model.run(
            vec![(in_id, tensor.view().into())],
            out_ids,
            None,
        )?;

        if outputs.is_empty() {
            return Err(EngineError::OutputError("Model produced no outputs".into()));
        }

        convert_output_to_array(&outputs[0])
    }

    pub fn run_with_inputs(
        &mut self,
        inputs_map: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        let mut rten_tensors = Vec::new();

        for (name, input) in &inputs_map {
            let node_id = if let Some(id) = self.model.find_node(name) {
                id
            } else if inputs_map.len() == 1 && !self.model.input_ids().is_empty() {
                self.model.input_ids()[0]
            } else {
                return Err(EngineError::OutputError(format!(
                    "Input node '{}' not found in model",
                    name
                )));
            };

            let shape: Vec<usize> = input.shape().to_vec();
            let data: Vec<f32> = if let Some(slice) = input.as_slice() {
                slice.to_vec()
            } else {
                input.iter().copied().collect()
            };
            rten_tensors.push((node_id, RtenTensor::<f32>::from_data(&shape, data)));
        }

        let input_pairs: Vec<(rten::NodeId, rten::ValueOrView)> = rten_tensors
            .iter()
            .map(|(node_id, tensor)| (*node_id, tensor.view().into()))
            .collect();

        let out_ids = self.model.output_ids();
        let outputs = self.model.run(input_pairs, out_ids, None)?;

        let mut results = HashMap::new();
        for (i, out_id) in out_ids.iter().enumerate() {
            let name = self
                .model
                .node_info(*out_id)
                .and_then(|info| info.name().map(|s| s.to_string()))
                .unwrap_or_else(|| format!("output_{}", i));
            if let Some(out_val) = outputs.get(i) {
                let array = convert_output_to_array(out_val)?;
                results.insert(name, array);
            }
        }

        Ok(results)
    }

    pub fn run_all(
        &mut self,
        input: ArrayD<f32>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        let shape: Vec<usize> = input.shape().to_vec();
        let data: Vec<f32> = if let Some(slice) = input.as_slice() {
            slice.to_vec()
        } else {
            input.iter().copied().collect()
        };

        let tensor = RtenTensor::<f32>::from_data(&shape, data);
        let in_ids = self.model.input_ids();
        if in_ids.is_empty() {
            return Err(EngineError::OutputError("Model has no inputs".into()));
        }
        let in_id = in_ids[0];

        let out_ids = self.model.output_ids();
        let outputs = self.model.run(
            vec![(in_id, tensor.view().into())],
            out_ids,
            None,
        )?;

        let mut results = HashMap::new();
        for (i, out_id) in out_ids.iter().enumerate() {
            let name = self
                .model
                .node_info(*out_id)
                .and_then(|info| info.name().map(|s| s.to_string()))
                .unwrap_or_else(|| format!("output_{}", i));
            if let Some(out_val) = outputs.get(i) {
                let array = convert_output_to_array(out_val)?;
                results.insert(name, array);
            }
        }

        Ok(results)
    }

    pub fn get_character_list(&self, _key: &str) -> Option<Vec<String>> {
        None
    }

    pub fn have_key(&self, _key: &str) -> bool {
        false
    }

    pub fn metadata_value(&self, key: &str) -> Option<String> {
        self.model.metadata().get(key).map(str::to_string)
    }
}

impl InferenceSession for RtenSession {
    fn run(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>, EngineError> {
        self.run(input)
    }

    fn run_all(&mut self, input: ArrayD<f32>) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_all(input)
    }

    fn run_with_inputs(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<String, ArrayD<f32>>, EngineError> {
        self.run_with_inputs(inputs)
    }

    fn get_character_list(&self, key: &str) -> Option<Vec<String>> {
        self.get_character_list(key)
    }

    fn have_key(&self, key: &str) -> bool {
        self.have_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_stage_session_reports_missing_path() {
        let err = load_stage_session("layout", Path::new("models/DocOCR/nope.rten"), &EngineConfig::default()).err().unwrap();
        let msg = err.to_string();
        assert!(msg.contains("layout model not found") && msg.contains("nope.rten"), "{msg}");
    }

    #[test]
    fn load_stage_session_rejects_unknown_extension() {
        let p = std::env::temp_dir().join("rusto-stage-test-model.bin");
        std::fs::write(&p, b"x").unwrap();
        let msg = load_stage_session("table", &p, &EngineConfig::default()).err().unwrap().to_string();
        let _ = std::fs::remove_file(&p);
        assert!(msg.contains("table model must be a .rten or .onnx file"), "{msg}");
    }
}
