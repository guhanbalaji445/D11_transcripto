


## Practical: Building our ASR System

We'll fine-tune `facebook/wav2vec2-base-960h`, a popular pre-trained model, on a small subset of the LibriSpeech dataset.

### 3.1. Setup and Dependencies

First, install the necessary libraries:

```bash
pip install torch torchaudio
pip install datasets transformers jiwer accelerate
# On some systems, you might need to install libsndfile for audio processing
# sudo apt-get install libsndfile1 (Linux)
# brew install libsndfile (macOS)
```
### 3.2. Choosing a Dataset

We'll use a very small part of the LibriSpeech dataset, made easily accessible by Hugging Face `datasets`. LibriSpeech contains read English speech.

```python
from datasets import load_dataset, Audio

# Load a small portion of LibriSpeech (e.g., first 50 examples from 'train.clean.100')
# Using a very small subset for quick demonstration
try:
    librispeech_full = load_dataset("librispeech_asr", "clean", split="train.100")
    # For a *really* quick demo, take only a few samples
    # For actual training, you'd want much more, e.g., split="train.clean.100"
    # For this demo, let's use even fewer samples to make it runnable quickly
    raw_dataset = librispeech_full.select(range(20)) # Using only 20 samples for speed
    print(f"Loaded {len(raw_dataset)} samples.")
except Exception as e:
    print(f"Failed to load LibriSpeech: {e}")
    print("Attempting to load 'PolyAI/minds14' as an alternative small dataset...")
    # Fallback to a different small dataset if LibriSpeech fails (e.g. due to size/network)
    # MINDS-14 is intent classification but has audio and transcriptions
    minds_full = load_dataset("PolyAI/minds14", name="en-US", split="train")
    raw_dataset = minds_full.select(range(20)) # Using only 20 samples for speed
    # We need to rename columns to match what Wav2Vec2 expects or what we use below
    raw_dataset = raw_dataset.rename_column("english_transcription", "text")
    raw_dataset = raw_dataset.map(lambda x: {"file": x["path"]}, batched=False)
    print(f"Loaded {len(raw_dataset)} samples from MINDS-14.")


# Let's see an example
print("\nSample from dataset:")
sample = raw_dataset[0]
print(sample)
print(f"Audio path: {sample['file']}")
print(f"Transcription: {sample['text']}")
print(f"Audio data: {sample['audio']}") # keys: 'path', 'array', 'sampling_rate'
```

### 3.3. Data Preprocessing

#### Audio Preprocessing
Wav2Vec2 expects audio inputs to be single-channel (mono) and have a sampling rate of 16kHz. The `datasets` library can handle resampling automatically. We also need a feature extractor to process the raw audio.

```python
from transformers import Wav2Vec2FeatureExtractor

model_checkpoint = "facebook/wav2vec2-base-960h"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)

# Ensure audio is 16kHz
raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_audio(batch):
    # The feature_extractor processes the audio array
    # It normalizes, pads/truncates, and converts to the model's expected format
    audio_input = batch["audio"]
    batch["input_values"] = feature_extractor(audio_input["array"], sampling_rate=audio_input["sampling_rate"]).input_values[0]
    # We also need the length for CTC loss, though Trainer handles it if not provided explicitly for input_values
    batch["input_length"] = len(batch["input_values"])
    return batch

# Apply audio preprocessing
# Note: For large datasets, use .map() with batched=True and multiprocessing
# For this tiny demo, batched=False is fine
processed_dataset_audio = raw_dataset.map(prepare_audio, remove_columns=raw_dataset.column_names)
print("\nSample after audio preprocessing:")
print(processed_dataset_audio[0].keys())
print(f"Shape of input_values: {len(processed_dataset_audio[0]['input_values'])}")
```

#### Text Preprocessing (Tokenization)
We need to convert text transcriptions into sequences of token IDs. For Wav2Vec2 fine-tuned with CTC, this typically involves character-level tokenization.

```python
from transformers import Wav2Vec2CTCTokenizer
import re

# 1. Create vocabulary
# For simplicity with a small dataset, we'll extract vocab from our small dataset.
# In practice, you'd use a pre-defined vocab or one from a larger text corpus.
def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  # Remove special characters, keep letters, numbers, and space
  # You might want to customize this regex for your specific dataset
  # Wav2Vec2's original tokenizer is more sophisticated (handles case, etc.)
  cleaned_text = re.sub(r"[^a-zA-Z0-9\s']", "", all_text).lower()
  vocab = list(set(cleaned_text))
  return {"vocab": [vocab], "all_text": [all_text]}

# For a tiny dataset, this vocab will be very small.
# It's better to use the pre-trained tokenizer's vocab if possible,
# or a vocab from a larger text corpus.
# For this demo, let's build a small vocab from our limited data.
vocabs = raw_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=raw_dataset.column_names)
# Consolidate vocab from all batches (if any)
dataset_vocab_list = list(set(c for v_list in vocabs["vocab"] for c in v_list))
vocab_dict = {v: k for k, v in enumerate(sorted(dataset_vocab_list))}
print(f"\nDataset Char Vocab: {vocab_dict}")

# Add CTC blank token, unknown token, and padding token
# The order matters for some pre-trained tokenizers.
# Wav2Vec2CTCTokenizer handles this internally when creating from scratch or loading.
vocab_dict["[UNK]"] = len(vocab_dict) # Unknown
vocab_dict["[PAD]"] = len(vocab_dict) # Padding

# Create a tokenizer instance
# For a robust system, save and load this vocab_dict as a JSON file
# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
# If you were using the vocab from the pre-trained model:
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_checkpoint)
print(f"\nPre-trained Tokenizer Vocab size: {tokenizer.vocab_size}")


# 2. Tokenize text
def prepare_labels(batch):
    # Normalize text (example: lowercase)
    # The tokenizer often handles normalization.
    target_transcription = batch["text"].lower() # Example normalization
    # Tokenize
    batch["labels"] = tokenizer(target_transcription).input_ids
    return batch

# Combine audio and text processing
def prepare_dataset_full(batch):
    # Audio
    audio_input = batch["audio"]
    batch["input_values"] = feature_extractor(audio_input["array"], sampling_rate=audio_input["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"]) # Optional, useful for some collators

    # Text
    # Normalize text to match tokenizer's expectations (e.g. lowercase, remove punctuation)
    # This step is crucial and depends on your specific data and tokenizer.
    # Wav2Vec2's default tokenizer expects uppercase and no special chars other than ' and space.
    # For simplicity, we'll lowercase and remove some punctuation.
    # A more robust approach involves careful text normalization.
    text = batch["text"].upper() # Wav2Vec2 tokenizer was trained on uppercase
    text = re.sub(r"[^\w\s']", "", text) # Keep alphanumeric, space, apostrophe

    batch["labels"] = tokenizer(text).input_ids
    return batch


# Apply combined preprocessing
# We use raw_dataset because it still has the 'text' and 'audio' columns
full_processed_dataset = raw_dataset.map(
    prepare_dataset_full,
    remove_columns=raw_dataset.column_names,
    num_proc=1 # Set to more cores if your dataset is large
)

print("\nSample after full preprocessing:")
print(full_processed_dataset[0].keys())
print(f"Labels (token IDs): {full_processed_dataset[0]['labels']}")
# Decode to verify
decoded_labels = tokenizer.decode(full_processed_dataset[0]['labels'])
print(f"Decoded labels: {decoded_labels}")
print(f"Original text: {raw_dataset[0]['text'].upper()}") # Compare with normalized original
```

### 3.4. The Model: Wav2Vec2

Wav2Vec2 is a model pre-trained on a massive amount of unlabeled audio data. It learns powerful representations of speech. We then fine-tune it on a smaller, labeled dataset for ASR.

*   **CNN Feature Extractor:** Processes raw audio into latent speech representations (these are initial embeddings).
*   **Transformer Encoder:** Takes these latent representations and applies self-attention to build contextualized representations.
*   **CTC Head:** A linear layer on top of the Transformer encoder predicts character probabilities for each time step. CTC (Connectionist Temporal Classification) loss is used to train the model to map audio sequences to text sequences of varying lengths.

```python
from transformers import Wav2Vec2ForCTC

# Load the pre-trained model, but configure it for our specific tokenizer/vocabulary for fine-tuning
# The vocab_size needs to match our tokenizer
model = Wav2Vec2ForCTC.from_pretrained(
    model_checkpoint,
    ctc_loss_reduction="mean",  # Recommended for stability
    pad_token_id=tokenizer.pad_token_id,
    vocab_size=tokenizer.vocab_size # Important!
)

# The feature extractor is part of the model (model.feature_extractor) but we defined it separately
# The model's `freeze_feature_encoder()` method can be used to prevent its weights from updating during fine-tuning
# This can be useful if you have a very small dataset, as the feature encoder is already well-trained.
model.freeze_feature_extractor()
```

### 3.5. Fine-Tuning the Model

We'll use the Hugging Face `Trainer` API for convenience. This requires a data collator.

```python
import torch
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric

# Data Collator: Pads input_values and labels dynamically per batch
# This is important because audio clips and transcriptions have varying lengths.
@dataclasses.dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor # Combines feature_extractor and tokenizer
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods.
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad audio inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

# Need a processor that combines feature_extractor and tokenizer
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Evaluation Metric: Word Error Rate (WER)
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 in labels (used for padding) as tokenizer expects valid token IDs
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids)
    # We do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training Arguments
# Using very few steps for demo purposes. Increase for real training.
# The output_dir will store model checkpoints.
training_args = TrainingArguments(
  output_dir="./wav2vec2-finetuned-librispeech-demo",
  group_by_length=True, # Speeds up training by grouping samples of similar input length
  per_device_train_batch_size=2, # Reduce if OOM, increase if GPU memory allows
  per_device_eval_batch_size=2,
  evaluation_strategy="steps", # Evaluate during training
  eval_steps=10, # Evaluate every N steps (very frequent for demo)
  logging_steps=5, # Log every N steps
  learning_rate=3e-4, # Standard learning rate for Wav2Vec2 fine-tuning
  warmup_steps=10, # Warmup steps (small for demo)
  save_steps=20, # Save checkpoint every N steps
  save_total_limit=2, # Only keep the last 2 checkpoints
  num_train_epochs=3,  # Number of epochs (very few for demo)
  # For actual training, consider more epochs (e.g., 10-30) and more data.
  # fp16=True, # Uncomment if you have a GPU that supports mixed-precision training
  push_to_hub=False, # Set to True if you want to upload to Hugging Face Hub
)

# Create a small train/test split from our tiny dataset for demonstration
# In a real scenario, you'd have a predefined, larger validation set.
if len(full_processed_dataset) > 5: # Ensure enough samples for a split
    split_dataset = full_processed_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
else: # If too few samples, use the whole dataset for both (not ideal, just for demo)
    train_dataset = full_processed_dataset
    eval_dataset = full_processed_dataset


# Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # Use a proper validation set
    tokenizer=processor.feature_extractor, # Important for the trainer to handle feature extraction if needed
)

# Start Fine-tuning!
print("\nStarting fine-tuning...")
# This will take some time, even with a small dataset.
# On CPU, it will be very slow. A GPU is highly recommended.
try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")
    print("This is likely due to the very small dataset size or resource constraints.")
    print("For a real run, use a larger dataset and ensure sufficient compute resources (GPU recommended).")

# Save the final model and processor
model_save_path = "./my_finetuned_asr_model"
processor_save_path = "./my_finetuned_asr_processor" # For the processor

if hasattr(trainer, 'is_world_process_zero') and trainer.is_world_process_zero():
    print(f"Saving model to {model_save_path}")
    model.save_pretrained(model_save_path)
    print(f"Saving processor to {processor_save_path}")
    processor.save_pretrained(processor_save_path)
elif not hasattr(trainer, 'is_world_process_zero'): # single process
    print(f"Saving model to {model_save_path}")
    model.save_pretrained(model_save_path)
    print(f"Saving processor to {processor_save_path}")
    processor.save_pretrained(processor_save_path)

```
**Note on Training:** With only 20 samples and few epochs, the model won't learn much. The WER will likely be very high. This is purely for demonstrating the pipeline. For meaningful results, you need hundreds or thousands of audio samples and more training epochs.

### 3.6. Inference and Evaluation

After training (or if you just want to test the pre-trained model or a saved fine-tuned one):

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# Load your fine-tuned model and processor
# If you didn't train, you can load the base pre-trained model
# model_path = "facebook/wav2vec2-base-960h" # For base model
# processor_path = "facebook/wav2Vec2-base-960h"
model_path = "./my_finetuned_asr_model" # Path to your saved fine-tuned model
processor_path = "./my_finetuned_asr_processor" # Path to your saved processor

try:
    loaded_model = Wav2Vec2ForCTC.from_pretrained(model_path)
    loaded_processor = Wav2Vec2Processor.from_pretrained(processor_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    print(f"\nLoaded fine-tuned model and processor from {model_path} and {processor_path}")

    # Take a sample from our original raw dataset for inference
    if raw_dataset:
        test_sample = raw_dataset[0] # Use the first sample
        audio_input_path = test_sample["audio"]["path"]
        reference_text = test_sample["text"]

        print(f"\nPerforming inference on sample: {audio_input_path}")
        print(f"Reference Text: {reference_text}")

        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_input_path)

        # Resample if necessary (Wav2Vec2 expects 16kHz)
        if sample_rate != loaded_processor.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=loaded_processor.feature_extractor.sampling_rate)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Process with feature extractor and tokenize
        input_values = loaded_processor(waveform.squeeze().numpy(),
                                   sampling_rate=loaded_processor.feature_extractor.sampling_rate,
                                   return_tensors="pt").input_values

        input_values = input_values.to(device)

        # Perform inference
        with torch.no_grad():
            logits = loaded_model(input_values).logits

        # Decode the output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = loaded_processor.batch_decode(predicted_ids)[0]

        print(f"Predicted Text: {transcription}")

        # Calculate WER for this single sample (optional)
        wer_score = wer_metric.compute(predictions=[transcription.upper()], references=[reference_text.upper()]) # Ensure case consistency for WER
        print(f"Word Error Rate (WER) for this sample: {wer_score:.4f}")
    else:
        print("raw_dataset is empty or not loaded, skipping inference example.")

except Exception as e:
    print(f"Could not load fine-tuned model/processor or run inference: {e}")
    print("This might be because training was skipped or failed, or paths are incorrect.")
    print("You can try inference with the base pre-trained model 'facebook/wav2vec2-base-960h' instead.")

```

---

## 4. Next Steps and Further Learning

*   **Train on More Data:** The key to a good ASR model is a large, diverse dataset. Try `librispeech_asr` with `train.clean.100` or `train.clean.360`.
*   **Experiment with Hyperparameters:** Learning rate, batch size, number of epochs, optimizer choices.
*   **Explore Different Pre-trained Models:**
    *   `wav2vec2-large-robust` (more robust to noise)
    *   `Hubert` (another self-supervised learning model)
    *   `Whisper` (OpenAI's powerful multilingual and multitasking model, often used for zero-shot or fine-tuning)
*   **Language Modeling:** Investigate how to integrate an external language model (e.g., using KenLM or an N-gram model) with the CTC decoder output to improve fluency. Libraries like `pyctcdecode` can help.
*   **Build from Scratch (Advanced):** Understand the PyTorch `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` to build the Transformer part yourself. This is a significant step up in complexity.
*   **Study CTC Loss:** Deeply understand how Connectionist Temporal Classification works.
*   **Other ASR Architectures:** Learn about RNN-Transducers, Listen Attend and Spell (LAS), etc.




**Key things for you to understand from this:**

1.  **The Pipeline:** Even with fine-tuning, there's a clear data flow: Audio -> Feature Extraction -> Transformer Processing -> Token ID Prediction -> Text.
2.  **Role of Pre-training:** Wav2Vec2's strength comes from its initial training on vast amounts of *unlabeled* audio. This allows it to learn fundamental acoustic properties. Fine-tuning adapts it to the specific task of *transcribing* labeled audio.
3.  **NLP in Action:** Text normalization and tokenization are essential for preparing target labels. The choice of tokenizer and vocabulary directly impacts the model.
4.  **Embeddings are Implicit and Explicit:**
    *   The CNNs in Wav2Vec2 create initial audio embeddings.
    *   The Transformer layers further refine these into contextualized embeddings.
    *   Positional embeddings are added to give sequence awareness.
5.  **Transformers for Context:** Self-attention within the Transformer encoder allows the model to weigh different parts of the audio input to understand each segment.
6.  **CTC for Alignment:** CTC loss is a clever way to train ASR models without needing frame-by-frame alignment between audio and text.

This example uses a very small dataset so it can run quickly for demonstration. Emphasize that real ASR requires much more data and compute.
