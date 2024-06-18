import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os
import sys
import tqdm
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
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            enroll_files = parts[3:]  # Enrollment file IDs
            embeddings = []
            for file in enroll_files:
                file_name = os.path.join(enroll_path, file)
                file_name += ".wav"
                audio, _ = sf.read(file_name)
                data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
                with torch.no_grad():
                    embedding = self.speaker_encoder.forward(data, aug=False)
                    embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)
            enrollments[model_id] = torch.mean(torch.stack(embeddings), dim=0)

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
        counter = 0  # Initialize the counter
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            test_file = parts[1]
            # Handle cases where trial_type is not present
            if len(parts) > 2:
                trial_type = parts[2]
                if trial_type in ['TC', 'TW']:
                    label = 1
                else:
                    label = 0
            else:
                label = None
            file_name = os.path.join(test_path, test_file)
            file_name += ".wav"
            audio, _ = sf.read(file_name)
            data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
            with torch.no_grad():
                test_embedding = self.speaker_encoder.forward(data, aug=False)
                test_embedding = F.normalize(test_embedding, p=2, dim=1)

            score = torch.mean(torch.matmul(test_embedding, enrollments[model_id].T)).detach().cpu().numpy()
            scores.append(score)
            if label is not None:
                labels.append(label)
            
            counter += 1  # Increment the counter
            if counter % 500000 == 0:
                print(f"Processed {counter} lines")
                # Print the score for each 10000 test file
                print(f"Score for model_id {model_id} and test_file {test_file}: {score}")

        # Compute EER and minDCF only if labels are available and compute_eer is True
        if compute_eer and labels:
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        else:
            EER = None
            minDCF = None

        # Write scores to answer.txt in the specified save_path
        answer_file_path = os.path.join(path_save_model, "answer.txt")
        with open(answer_file_path, 'w') as f:  # Append to the file
            for score in scores:
                f.write(f"{score}\n")

        # Zip the answer.txt file
        submission_zip_path = os.path.join(path_save_model, "submission.zip")
        with zipfile.ZipFile(submission_zip_path, 'w') as zipf:
            zipf.write(answer_file_path, os.path.basename(answer_file_path))

        return EER, minDCF, scores


    def test_network2(self, test_list, test_path, path_save_model, compute_eer=True):
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

        for line in lines:
            parts = line.split()
            model_ids.append(parts[0])
            test_files.append(parts[1])
            if len(parts) > 2:
                trial_type = parts[2]
                if trial_type in ['TC', 'TW']:
                    labels.append(1)
                else:
                    labels.append(0)

        audio_data = []
        for test_file in test_files:
            file_name = os.path.join(test_path, test_file)
            file_name += ".wav"
            audio, _ = sf.read(file_name)
            audio_data.append(torch.FloatTensor(np.stack([audio], axis=0)).cuda())

        audio_data = torch.cat(audio_data, dim=0)

        with torch.no_grad():
            test_embeddings = self.speaker_encoder.forward(audio_data, aug=False)
            test_embeddings = F.normalize(test_embeddings, p=2, dim=1)

        model_embeddings = torch.stack([enrollments[model_id] for model_id in model_ids]).cuda()
        scores = torch.mean(torch.matmul(test_embeddings, model_embeddings.T), dim=1).detach().cpu().numpy()

        if compute_eer and labels:
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        else:
            EER = None
            minDCF = None

        answer_file_path = os.path.join(path_save_model, "answer.txt")
        with open(answer_file_path, 'w') as f:
            for score in scores:
                f.write(f"{score}\n")

        submission_zip_path = os.path.join(path_save_model, "submission.zip")
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
