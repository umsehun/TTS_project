import time
import os
import random
import numpy as np
import torch
import torch.utils.data
import librosa
import json

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence


class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, data_config):
        self.audiopaths_and_text = self._load_metadata(audiopaths_and_text)
        self.data_config = data_config
        # 초기화 시점에 경로 검증
        self.validate_paths()

    def validate_paths(self):
        """데이터셋의 경로들을 검증"""
        for idx, item in enumerate(self.audiopaths_and_text):
            audio_path, text = item
            if not os.path.exists(audio_path):
                print(f"Invalid audio path at index {idx}: {audio_path}")

    def _load_metadata(self, path):
        """CSV 또는 JSON 파일에서 메타데이터를 로드"""
        all_data = []
        
        # CSV 파일 생성 (없는 경우)
        csv_path = path  # 직접 입력받은 경로 사용
        if not os.path.exists(csv_path):
            print("Creating metadata.csv from JSON files...")
            json_files = [
                os.path.join(os.path.dirname(csv_path), f"dataset_{i}.json") 
                for i in range(1, 6)  # 1부터 5까지
            ]
            
            with open(csv_path, 'w', encoding='utf-8') as f:
                for json_file in sorted(json_files):  # 파일명 순서대로 정렬
                    try:
                        if not os.path.exists(json_file):
                            print(f"File not found: {json_file}")
                            continue
                            
                        print(f"Processing {json_file}...")
                        with open(json_file, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                            for item in data:
                                # 절대 경로로 변환
                                audio_path = os.path.join(
                                    os.path.dirname(csv_path), 
                                    "wavs", 
                                    os.path.basename(item['audio_filepath'])
                                )
                                line = f"{audio_path}|{item['text']}\n"
                                f.write(line)
                        print(f"Completed processing {json_file}")
                    except Exception as e:
                        print(f"Error processing {json_file}: {e}")
                        continue
        
        # CSV 파일 읽기
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                all_data = [line.strip().split('|') for line in f]
            print(f"Total loaded items from {csv_path}: {len(all_data)}")
            print("First few entries:")
            for i in range(min(5, len(all_data))):
                print(f"Entry {i}: {all_data[i]}")
        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
            return []

        return all_data

    def get_audio_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text
        text = self.get_text(text)
        spec = self.get_spectrogram(audiopath)
        return (text, spec)

    def get_spectrogram(self, audiopath):
        """오디오 파일에서 스펙트로그램 생성"""
        audio, sr = librosa.load(audiopath, sr=self.data_config["sampling_rate"])
        
        # STFT 계산
        stft = librosa.stft(
            audio,
            n_fft=self.data_config["filter_length"],
            hop_length=self.data_config["hop_length"],
            win_length=self.data_config["win_length"],
        )
        
        # Mel-스펙트로그램으로 변환
        mel_filter = librosa.filters.mel(
            sr=sr,
            n_fft=self.data_config["filter_length"],
            n_mels=self.data_config["n_mel_channels"],
            fmin=self.data_config["mel_fmin"],
            fmax=self.data_config["mel_fmax"],
        )
        mel_spectrogram = np.dot(mel_filter, np.abs(stft))

        # 텐서로 변환
        mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
        return mel_spectrogram

    def get_text(self, text):
        """텍스트 전처리 및 정수 ID 시퀀스로 변환"""
        # 필요한 텍스트 클리너 적용
        return text_to_sequence(text, self.data_config["text_cleaners"])

    def get_text_data(self, index):
        try:
            # 데이터셋에서 텍스트 데이터를 가져옴
            text = self.audiopaths_and_text[index][1]  # 예: (audio_path, text)
            
            # 텍스트 유효성 검사
            if not isinstance(text, str):
                print(f"Warning: Text at index {index} is not string type: {type(text)}")
                text = str(text)
            
            if not text.strip():
                print(f"Warning: Empty text at index {index}")
                return None, None
            
            # 텍스트를 정수 시퀀스로 변환
            try:
                text = self.get_text(text)
                if not text:  # 변환 결과가 비어있는 경우
                    print(f"Warning: Text conversion resulted in empty sequence at index {index}")
                    return None, None
            except Exception as e:
                print(f"Error converting text at index {index}: {e}")
                print(f"Original text: {text}")
                return None, None
            
            text_length = len(text)
            
            # 기본적인 길이 검증만 수행
            if text_length < 1:
                print(f"Warning: Empty text sequence at index {index}")
                return None, None
            
            print(f"Successfully processed text at index {index}: length={text_length}")
            return text, text_length
            
        except Exception as e:
            print(f"Error processing text data at index {index}: {e}")
            print(f"Data at index: {self.audiopaths_and_text[index]}")
            return None, None

    def get_wav_data(self, index):
        try:
            audio_path = self.audiopaths_and_text[index][0]
            # librosa를 사용하여 오디오 파일 로드
            audio, sr = librosa.load(audio_path, sr=self.data_config["sampling_rate"])
            # 텐서로 변환
            wav = torch.FloatTensor(audio)
            wav_length = wav.size(0)
            
            # 디버깅을 위한 출력
            print(f"Successfully loaded audio at index {index}: {audio_path}")
            print(f"Audio shape: {wav.shape}, Sample rate: {sr}")
            
            return wav, wav_length
        except Exception as e:
            print(f"Error loading wav data for index {index}: {e}")
            print(f"Audio path: {audio_path}")
            return None, None

    def __getitem__(self, index):
        try:
            text, text_length = self.get_text_data(index)
            if text is None or text_length is None:
                print(f"Invalid text data at index {index}")
                return None

            audio_path = self.audiopaths_and_text[index][0]
            spectrogram = self.get_spectrogram(audio_path)
            spec_length = spectrogram.shape[1]

            wav, wav_length = self.get_wav_data(index)
            if wav is None or wav_length is None:
                print(f"Invalid audio data at index {index}")
                return None

            return text, text_length, spectrogram, spec_length, wav, wav_length
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            return None

    def __len__(self):
        return len(self.audiopaths_and_text)

    # 예제 디버깅 코드
    def debug_print(self):
        for i, data in enumerate(self.audiopaths_and_text):
            print(f"Data at index {i}: {data}")


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        try:
            # None 값 필터링
            batch = [b for b in batch if b is not None]
            if not batch:
                raise ValueError("Empty batch after filtering None values")

            # 디버깅을 위한 배치 구조 출력
            print("Batch structure:")
            for i, item in enumerate(batch):
                shapes = []
                for x in item:
                    if isinstance(x, (torch.Tensor, np.ndarray)):
                        shapes.append(x.shape)
                    elif isinstance(x, (list, tuple, str)):
                        shapes.append(len(x))
                    else:
                        shapes.append(type(x))
                print(f"Item {i}: {shapes}")

            # wav 데이터 처리 수정
            for i in range(len(batch)):
                row = list(batch[i])
                wav = row[4]
                if wav.dim() == 1:  # 1차원 텐서인 경우
                    wav = wav.unsqueeze(0)  # (T) -> (1, T)
                row[4] = wav
                batch[i] = tuple(row)

            max_text_len = max([len(x[0]) for x in batch])
            max_spec_len = max([x[2].shape[1] for x in batch])
            max_wav_len = max([x[4].shape[1] if x[4].dim() > 1 else x[4].shape[0] for x in batch])

            text_lengths = torch.LongTensor(len(batch))
            spec_lengths = torch.LongTensor(len(batch))
            wav_lengths = torch.LongTensor(len(batch))

            text_padded = torch.LongTensor(len(batch), max_text_len)
            spec_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_spec_len)
            wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

            text_padded.zero_()
            spec_padded.zero_()
            wav_padded.zero_()

            for i in range(len(batch)):
                row = batch[i]
                text = row[0]
                text_padded[i, :len(text)] = torch.LongTensor(text)
                text_lengths[i] = len(text)

                spec = row[2]
                spec_padded[i, :, :spec.shape[1]] = spec
                spec_lengths[i] = spec.shape[1]

                wav = row[4]
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                wav_padded[i, :, :wav.shape[1]] = wav
                wav_lengths[i] = wav.shape[1]

            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths

        except Exception as e:
            print(f"Error in collate function: {str(e)}")
            raise e


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size

