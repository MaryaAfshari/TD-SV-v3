#Ya Latif
#Date: 18.3.1403 Khordad mah
#Date: 7.6.2024 June 
#Author: Maryam Afshari -Iranian
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import os
import tqdm
import pickle
import time
import sys
import numpy as np

from tools3 import *
from loss3 import AAMsoftmax
from model3 import ECAPA_TDNN, SEModule
from dataloader3 import phrases_to_phonemes

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
        self.se_modules = nn.ModuleList([SEModule(C) for _ in range(4)]).cuda()
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        self.phoneme_loss = nn.CrossEntropyLoss().cuda()

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model parameter number = %.2f" % (sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def forward(self, x, phoneme_ids, aug=False):
        x, phoneme_posterior = self.speaker_encoder(x, phoneme_ids, aug)
        for se_module in self.se_modules:
            x = se_module(x)
        return x, phoneme_posterior

    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, speaker_labels, phrase_labels, phoneme_ids) in enumerate(loader, start=1):
            self.zero_grad()
            speaker_labels = torch.LongTensor(speaker_labels).cuda()
            phrase_labels = torch.LongTensor(phrase_labels).cuda()
            phoneme_ids = torch.LongTensor(phoneme_ids).cuda()
            speaker_embedding, phoneme_posterior = self.forward(data.cuda(), phoneme_ids, aug=True)
            nloss, prec = self.speaker_loss(speaker_embedding, speaker_labels)
            phoneme_loss = self.phoneme_loss(phoneme_posterior, phrase_labels)
            total_loss = nloss + 0.3 * phoneme_loss
            total_loss.backward()
            self.optim.step()
            index += len(speaker_labels)
            top1 += prec
            loss += total_loss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + " [%2d] Lr: %5f, Training: %.2f%%, Loss: %.5f, ACC: %2.2f%% \r" % (epoch, lr, 100 * (num / loader.__len__()), loss / (num), top1 / index * len(speaker_labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(speaker_labels)

    def eval_network(self, eval_list, eval_path):
        print("Evaluating network...")
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            file_name = os.path.join(eval_path, file) + ".wav"
            audio, _ = sf.read(file_name)
            data_1 = torch.FloatTensor(np.stack([audio], axis=0)).cuda()

            # Splitted utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            feats = np.stack(feats, axis=0).astype(np.float)
            data_2 = torch.FloatTensor(feats).cuda()

            # Speaker embeddings
            with torch.no_grad():
                embedding_1, _ = self.forward(data_1, phoneme_ids=None, aug=False)  # phoneme_ids=None چون در ارزیابی نیازی نیست
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2, _ = self.forward(data_2, phoneme_ids=None, aug=False)  # phoneme_ids=None چون در ارزیابی نیازی نیست
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []
        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # Compute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def enroll_network(self, enroll_list, enroll_path, path_save_model):
        print("Enrolling network...")
        self.eval()
        enrollments = {}
        lines = open(enroll_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            phrase_id = parts[1]  # Extract phrase_id
            enroll_files = parts[4:]  # Enrollment file IDs
            embeddings = []
            for file in enroll_files:
                file_name = os.path.join(enroll_path, file) + ".wav"
                audio, _ = sf.read(file_name)
                data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
                phoneme_ids = torch.LongTensor(phrases_to_phonemes[phrase_id]).cuda()  # Get phoneme ids
                with torch.no_grad():
                    embedding, _ = self.forward(data, phoneme_ids, aug=False)
                    embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)
            enrollments[(model_id, phrase_id)] = torch.mean(torch.stack(embeddings), dim=0)  # Use tuple (model_id, phrase_id)

        # Ensure the directory exists
        os.makedirs(path_save_model, exist_ok=True)

        # Save enrollments using the provided path
        with open(os.path.join(path_save_model, "enrollments.pkl"), "wb") as f:
            pickle.dump(enrollments, f)

    def test_network(self, test_list, test_path, path_save_model):
        print("Testing network...")
        self.eval()
        # Load enrollments
        enrollments_path = os.path.join(path_save_model, "enrollments.pkl")
        print(f"Loading enrollments from {enrollments_path}")
        with open(enrollments_path, "rb") as f:
            enrollments = pickle.load(f)

        scores, labels = [], []
        lines = open(test_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            test_file = parts[1]
            trial_type = parts[2]
            phrase_id = parts[3]  # Extract phrase_id
            # Assign labels based on trial-type
            label = 1 if trial_type in ['TC', 'TW'] else 0
            file_name = os.path.join(test_path, test_file) + ".wav"
            audio, _ = sf.read(file_name)
            data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
            phoneme_ids = torch.LongTensor(phrases_to_phonemes[phrase_id]).cuda()  # Get phoneme ids
            with torch.no_grad():
                test_embedding, _ = self.forward(data, phoneme_ids, aug=False)
                test_embedding = F.normalize(test_embedding, p=2, dim=1)

            # Use tuple (model_id, phrase_id) to get the correct enrollment
            enrollment = enrollments.get((model_id, phrase_id))
            if enrollment is not None:
                score = torch.mean(torch.matmul(test_embedding, enrollment.T)).detach().cpu().numpy()
                scores.append(score)
                labels.append(label)
            else:
                print(f"Enrollment not found for model {model_id} with phrase {phrase_id}")

        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

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
