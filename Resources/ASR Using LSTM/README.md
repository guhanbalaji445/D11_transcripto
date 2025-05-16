# LSTM for ASR (Automatic Speech Recognition)

## What is LSTM?

LSTM (Long Short-Term Memory) is a special type of Recurrent Neural Network (RNN) that is good at learning sequences (text, speech, audio, etc), especially when long-range dependencies are involved.

### Why LSTM?

Regular RNNs struggle with long sequences because of the vanishing/exploding gradient problem. That means they "forget" things from earlier in the sequence.

LSTM solves this using a clever structure that allows it to remember information for a long time. Each LSTM unit has a cell-state (memory) and 3 gates:

1. **Forget gate**: Decides what is not important, so that it can "forget".
2. **Input gate**: Decides what new info it can add from other units.
3. **Output gate**: Decides what to output to other units.

At each timestep in a sequence:

1. The LSTM looks at the current input and the previous hidden state.
2. It updates its cell state using the gates.
3. It outputs a new hidden state, which goes to the next timestep.

Video reference: [LSTM Explained](https://youtu.be/YCzL96nL7j0?si=12raVcUGHhW_ebc8)

---

## ASR Using LSTM

Refer to the `.ipynb` file in the folder. Use that as a template and improve on it.

### Dataset Used

```python
dataset = torchaudio.datasets.LIBRISPEECH(root="./data", url="train-clean-100", download=True)
```

* Contains approx. \~100h of speech
* Sampled at 16kHz
* Stored as `.wav` files

Example:

```python
waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = dataset[1]
print("Waveform shape:", waveform.shape)
print("Sample rate:", sample_rate)
print("Transcript:", transcript)
```

Output:

```
Waveform shape: torch.Size([1, 255120])  # 255120 -> number of samples
Sample rate: 16000
Transcript: THAT HAD ITS SOURCE AWAY BACK IN THE WOODS OF THE OLD CUTHBERT PLACE...
```

---

## Data Processing Pipeline

Raw audio waveforms are converted into Mel spectrograms using torchaudio.transforms.MelSpectrogram. This transforms the 1D waveform into a 2D time-frequency representation, which is more suitable for speech models. A log transform is applied to the spectrogram to compress dynamic ranges and improve learning.

```python
import string

# Spectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=80
)

def extract_features(waveform):
    mel = mel_transform(waveform).clamp(min=1e-5).log()
    return mel.squeeze(0).transpose(0, 1)  # shape: [time_steps, features]
```

Custom Dataset Class (ASRDataset):
- Loads waveform and transcript pairs from the LibriSpeech dataset.
- Ensures audio is resampled to 16 kHz if needed.
- Extracts Mel spectrogram features.
- Converts transcripts to sequences of character indices (targets).

```python
class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, torchaudio_dataset):
        self.data = torchaudio_dataset

    def __getitem__(self, idx):
        waveform, sr, transcript, *_ = self.data[idx]
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        
        features = extract_features(waveform)
        targets = torch.tensor(text_to_indices(transcript), dtype=torch.long)
        return features, targets

    def __len__(self):
        return len(self.data)
```
Collation Function (collate_fn):
- Pads feature sequences in a batch to the same length for batch processing.
- Concatenates targets for use with Connectionist Temporal Classification (CTC) loss.
- Returns padded inputs, targets, and their respective lengths.

```python

def collate_fn(batch):
    inputs, targets = zip(*batch)

    input_lengths = torch.tensor([x.shape[0] for x in inputs])
    target_lengths = torch.tensor([y.shape[0] for y in targets])

    # Pad inputs
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs,batch_first=True)
    padded_targets = torch.cat(targets)  # Targets are concatenated for CTC loss

    return padded_inputs, padded_targets, input_lengths, target_lengths
```

```python
asr_dataset = ASRDataset(dataset)

data_loader = torch.utils.data.DataLoader(
    asr_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)
```
### LSTM Model

Here is an example of what you could do.
- B - Batch size (8 in this case)
- T - Time steps (number of frames in each clip)
- F - Number of input features (80 Mel-frequency features)
- H - Number of hidden LSTM units (256x2 since bidirectional)
```python
class ASRModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, output_dim=len(char2idx), num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)  # bidirectional

    def forward(self, x):
        outputs, _ = self.lstm(x)           # x: [B, T, F] → outputs: [B, T, 2*H]
        logits = self.classifier(outputs)   # [B, T, vocab_size]
        return logits.log_softmax(dim=-1)   # CTC expects log-probabilities
```

Refer to the ipynb file on how you could set up a training loop and how to check Word Error Rate (WER) and Character Error Rate (CER)

Note that, the training loop in the ipynb file does it for a smaller subset of dataset. Training on the full dataset will take ~3h and give around ~40% WER.

## Beam Search Decoding (with Language Model)

Beam search is a decoding algorithm used to find the most likely transcription from model outputs (like CTC logits).

- At each time step, it keeps the **top-k (beam width)** most likely sequences instead of just the single best one.
- It considers both the **acoustic score** (from the ASR model) and optionally a **language model score** to rank hypotheses.
$$
\text{Combined Score} = \alpha \cdot \text{Acoustic Score} + \beta \cdot \text{LM Score}
$$
- $\alpha$ and $\beta$ can be tuned to choose the best fit.
- This improves over greedy decoding (choosing the single highest probability for each word), by exploring multiple paths and combining the best overall result.

### Example:
- Beam width = 3  (beam width is tunable)
- At each time step, keep the 3 best hypotheses.
- At the end, return the highest-scoring complete sequence.

This helps avoid mistakes where the model picks a wrong character early and gets stuck with it.

## What is an LM and how to set it up?

A n-gram Language Model assigns probabilities to sequences of words/characters (of length n), based on how they form meaningful sentences.

To use a pre-trained LM, download an LM for speech (eg. LibriSpeech 3-gram), download the `.txt` or `.arpa` file and import it into your `.ipynb` notebook as a dataset.


For example:
```python
!pip install https://github.com/kpu/kenlm/archive/master.zip
import kenlm
lm = kenlm.Model('/kaggle/input/lm-for-asr/3-gram.pruned.3e-7.arpa')
```

# Improvements
- Train on the entire dataset (100h) which will take a few hours on GPU, for about 30 epochs.
- Experiment with the ASR architecture.
- Tune `beam_width` as well as `alpha` and `beta` values to achieve a good WER (<40%).
- Train on an Indian English dataset and experiment with the model.
- If any doubts, please ask :)
