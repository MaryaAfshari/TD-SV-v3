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

    def enroll_network(self, enroll_list, enroll_path, path_save_model):
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
                all_audio_data.append(torch.FloatTensor(np.stack([audio], axis=0)).cuda())
                file_to_index[file] = index
                index += 1

        all_audio_data = torch.cat(all_audio_data, dim=0)

        with torch.no_grad():
            all_embeddings = self.speaker_encoder.forward(all_audio_data, aug=False)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

        # Compute mean embeddings for each model
        for model_id, files in model_files_dict.items():
            indices = [file_to_index[file] for file in files]
            model_embeddings = all_embeddings[indices]
            enrollments[model_id] = torch.mean(model_embeddings, dim=0)

        os.makedirs(path_save_model, exist_ok=True)
        with open(os.path.join(path_save_model, "enrollments.pkl"), "wb") as f:
            pickle.dump(enrollments, f)

    def test_network(self, test_list, test_path, path_save_model, compute_eer=True):
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

        # Extract embeddings for all unique test files
        audio_data = []
        for test_file in test_files:
            file_name = os.path.join(test_path, test_file) + ".wav"
            audio, _ = sf.read(file_name)
            audio_data.append(torch.FloatTensor(np.stack([audio], axis=0)).cuda())

        audio_data = torch.cat(audio_data, dim=0)

        with torch.no_grad():
            test_embeddings = self.speaker_encoder.forward(audio_data, aug=False)
            test_embeddings = F.normalize(test_embeddings, p=2, dim=1)

        test_embedding_dict = {test_file: test_embeddings[i] for i, test_file in enumerate(test_files)}

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
