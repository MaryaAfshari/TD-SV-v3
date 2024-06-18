import glob, numpy, os, random, soundfile, torch
from scipy import signal

class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        
        # Load data & labels
        self.data_list = []
        self.speaker_labels = []
        self.phrase_labels = []  # New phrase labels
        
        lines = open(train_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        dictkeys = list(set([x.split()[1] for x in lines]))  # Changed to index 1 for speaker-id
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        
        for index, line in enumerate(lines):
            try:
                speaker_label = dictkeys[line.split()[1]]  # Changed to index 1 for speaker-id
                #phrase_label = int(line.split()[2])  # Assuming phrase ID is the third column
                phrase_label = line.split()[2]  # Treat phrase ID as a string
                file_name = os.path.join(train_path, line.split()[0])  # Changed to index 0 for train-file-id
                file_name += ".wav"  # I added for making a path 18-2-1403 Ordibehesht - May
                
                self.speaker_labels.append(speaker_label)
                self.phrase_labels.append(phrase_label)
                self.data_list.append(file_name)
                #print(f"Speaker Label: {speaker_label}, Phrase Label: {phrase_label}, File Name: {file_name}")
            except ValueError:
                print(f"Skipping line with non-integer phrase ID: {line}")

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        file_name = self.data_list[index]
        #print(f"Reading file: {file_name}")
        audio, sr = soundfile.read(file_name)
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        
        # Data Augmentation
        augtype = 0  # I have changed it at 18-2-1403 Ordibehesht - May
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        
        return torch.FloatTensor(audio[0]), self.speaker_labels[index], self.phrase_labels[index]

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
