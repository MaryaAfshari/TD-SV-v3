import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os
import sys
import time
import pickle
import zipfile

from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()

        ## Classifier
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        print("I am in the train network ....")
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        print("Loader Length = ", loader.__len__())

        for num, (data, speaker_labels, phrase_labels) in enumerate(loader, start=1):
            self.zero_grad()
            speaker_labels = torch.LongTensor(speaker_labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, speaker_labels)
            nloss.backward()
            self.optim.step()

            index += len(speaker_labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(speaker_labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(speaker_labels)

    def enroll_network(self, enroll_list, enroll_path, path_save_model, batch_size=1):
        self.eval()
        print("I am in enroll method ....")
        enrollments = {}
        lines = open(enroll_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        model_files_dict = {}

        # Collect files for each model
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            enroll_files = parts[3:]  # Enrollment file IDs
            if model_id not in model_files_dict:
                model_files_dict[model_id] = []
            model_files_dict[model_id].extend(enroll_files)

        all_audio_data = []
        file_to_index = {}
        index = 0

        # Load and store all unique audio files
        for model_id, files in model_files_dict.items():
            unique_files = list(set(files))
            for file in unique_files:
                file_name = os.path.join(enroll_path, file) + ".wav"
                audio, _ = sf.read(file_name)
                audio_tensor = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
                all_audio_data.append(audio_tensor)
                #all_audio_data.append(torch.FloatTensor(np.stack([audio], axis=0)).cuda())
                file_to_index[file] = index
                index += 1
                print(f"Loaded tensor shape: {audio_tensor.shape} for file {file_name}")

        # Find the maximum size for padding
        max_size = max(tensor.size(1) for tensor in all_audio_data)
        print(f"Max size for padding: {max_size}")

        # Pad tensors to the same size
        padded_audio_data = []
        for tensor in all_audio_data:
            if tensor.size(1) < max_size:
                pad_size = max_size - tensor.size(1)
                padded_tensor = F.pad(tensor, (0, pad_size))
                padded_audio_data.append(padded_tensor)
                print(f"Padded tensor from shape {tensor.shape} to {padded_tensor.shape}")
            else:
                padded_audio_data.append(tensor)

        #all_audio_data = torch.cat(all_audio_data, dim=0)

        all_audio_data = torch.cat(padded_audio_data, dim=0)
        print(f"All audio data shape after padding and concatenation: {all_audio_data.shape}")

        print("I am in the enroll network after padding and  brfore affecting on the matrix ....")

        # with torch.no_grad():
        #     all_embeddings = self.speaker_encoder.forward(all_audio_data, aug=False)
        #     all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        all_embeddings = []
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(padded_audio_data), batch_size):
                batch = padded_audio_data[i:i+batch_size]
                batch_data = torch.cat(batch, dim=0)
                print(f"Processing batch {i//batch_size + 1} with shape: {batch_data.shape}")
                batch_embeddings = self.speaker_encoder.forward(batch_data, aug=False)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(batch_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"All embeddings shape: {all_embeddings.shape}")

        print("I am in the enroll network after affecting on the matrix ....")

        # Compute mean embeddings for each model
        for model_id, files in model_files_dict.items():
            indices = [file_to_index[file] for file in files]
            model_embeddings = all_embeddings[indices]
            enrollments[model_id] = torch.mean(model_embeddings, dim=0)

        os.makedirs(path_save_model, exist_ok=True)
        with open(os.path.join(path_save_model, "enrollments.pkl"), "wb") as f:
            pickle.dump(enrollments, f)

    def test_network(self, test_list, test_path, path_save_model, compute_eer=True, batch_size=1):
        print("I am in the test method ....")
        self.eval()
        enrollments_path = os.path.join(path_save_model, "enrollments.pkl")
        print(f"Loading enrollments from {enrollments_path}")
        with open(enrollments_path, "rb") as f:
            enrollments = pickle.load(f)

        scores, labels = [], []
        lines = open(test_list).read().splitlines()
        lines = lines[1:]  # Skip the header row

        model_ids = []
        test_files = []
        label_dict = {}

        for line in lines:
            parts = line.split()
            model_id = parts[0]
            test_file = parts[1]
            if len(parts) > 2:
                trial_type = parts[2]
                label = 1 if trial_type in ['TC', 'TW'] else 0
                label_dict[test_file] = label
            model_ids.append(model_id)
            test_files.append(test_file)

        test_files = list(set(test_files))  # Unique test files

        # Process test files in batches
        max_size = 0
        audio_tensors = []

        for i in range(0, len(test_files), batch_size):
            batch_files = test_files[i:i + batch_size]
            batch_audio = []

            for test_file in batch_files:
                file_name = os.path.join(test_path, test_file) + ".wav"
                audio, _ = sf.read(file_name)
                audio_tensor = torch.FloatTensor(np.stack([audio], axis=0))
                batch_audio.append(audio_tensor)
                if audio_tensor.size(1) > max_size:
                    max_size = audio_tensor.size(1)
                print(f"Loaded tensor shape: {audio_tensor.shape} for file {file_name}")

            print(f"Max size for padding in batch {i//batch_size + 1}: {max_size}")

            # Pad and process batch audio
            padded_batch_audio = []
            for tensor in batch_audio:
                if tensor.size(1) < max_size:
                    pad_size = max_size - tensor.size(1)
                    padded_tensor = F.pad(tensor, (0, pad_size))
                    padded_batch_audio.append(padded_tensor)
                    print(f"Padded tensor from shape {tensor.shape} to {padded_tensor.shape}")
                else:
                    padded_batch_audio.append(tensor)

            batch_data = torch.cat(padded_batch_audio, dim=0).cuda()
            with torch.no_grad():
                batch_embeddings = self.speaker_encoder.forward(batch_data, aug=False)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                audio_tensors.extend(batch_embeddings)

            # Clear GPU cache
            del batch_data, batch_embeddings, padded_batch_audio
            torch.cuda.empty_cache()

        test_embeddings = torch.stack(audio_tensors)
        print(f"Test embeddings shape: {test_embeddings.shape}")
        
        test_embedding_dict = {test_files[i]: test_embeddings[i] for i in range(len(test_files))}

        # Compute scores
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            test_file = parts[1]
            score = torch.mean(torch.matmul(test_embedding_dict[test_file].unsqueeze(0), enrollments[model_id].T)).detach().cpu().numpy()
            scores.append(score)
            if test_file in label_dict:
                labels.append(label_dict[test_file])

        if compute_eer and labels:
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        else:
            EER = None
            minDCF = None

        answer_file_path = os.path.join(path_save_model, "answer_dev_try2.txt")
        with open(answer_file_path, 'w') as f:
            for score in scores:
                f.write(f"{score}\n")

        submission_zip_path = os.path.join(path_save_model, "submission_dev_try2.zip")
        with zipfile.ZipFile(submission_zip_path, 'w') as zipf:
            zipf.write(answer_file_path, os.path.basename(answer_file_path))

        return EER, minDCF, scores
    

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
