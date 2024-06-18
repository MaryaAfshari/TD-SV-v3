import argparse
import glob
import os
import torch
import warnings
import time
import soundfile as sf
import torch.nn.functional as F
import zipfile
import pickle
from tools import *
from dataLoader2 import train_loader
from ECAPAModel_mysystem import ECAPAModel
import numpy as np

def main():
    print("I am in the main")
    parser = argparse.ArgumentParser(description="ECAPA_trainer")

    # Training Settings
    parser.add_argument('--num_frames', type=int, default=200, help='Duration of the input segments, eg: 200 for 2 second')
    parser.add_argument('--max_epoch', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_cpu', type=int, default=1, help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=10, help='Test and save every [test_step] epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')

    # Paths
    parser.add_argument('--train_list', type=str, default="/mnt/disk1/data/TdSVC2024/task1/docs/train_labels.txt", help='The path of the training list')
    parser.add_argument('--train_path', type=str, default="/mnt/disk1/data/TdSVC2024/task1/wav/train", help='The path of the training data')
    parser.add_argument('--eval_list', type=str, default="/mnt/disk1/data/TdSVC2024/task1/docs/dev_trials.txt", help='The path of the evaluation list')
    parser.add_argument('--eval_path', type=str, default="/mnt/disk1/data/TdSVC2024/task1/wav/evaluation", help='The path of the evaluation data')
    parser.add_argument('--enroll_list', type=str, default="/mnt/disk1/data/TdSVC2024/task1/docs/dev_model_enrollment.txt", help='The path of the enrollment list')
    parser.add_argument('--enroll_path', type=str, default="/mnt/disk1/data/TdSVC2024/task1/wav/enrollment", help='The path of the enrollment data')
    parser.add_argument('--musan_path', type=str, default="/data08/Others/musan_split", help='The path to the MUSAN set')
    parser.add_argument('--rir_path', type=str, default="/data08/Others/RIRS_NOISES/simulated_rirs", help='The path to the RIR set')
    parser.add_argument('--save_path', type=str, default="/mnt/disk1/users/afshari/MyEcapaModelTry2", help='Path to save the score.txt and models')
    parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')
    parser.add_argument('--path_save_model', type=str, default="/mnt/disk1/users/afshari/MyEnrollmentTry2", help='Path to save the enrollment and models')

    # Model and Loss settings
    parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
    parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
    parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
    parser.add_argument('--n_class', type=int, default=1620, help='Number of speakers')

    # Command
    parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')
    parser.add_argument('--enroll', dest='enroll', action='store_true', help='Only do enrollment')
    parser.add_argument('--test', dest='test', action='store_true', help='Only do testing')

    # Initialization
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    args = init_args(args)

    print("Arguments parsed and initialized")

    # Define the data loader
    trainloader = train_loader(**vars(args))
    trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)

    print("Data loader initialized")

    # Search for the existing models
    modelfiles = glob.glob('%s/model_0*.model' % args.save_path)
    modelfiles.sort()

    # Load model
    if args.initial_model != "":
        print(f"Model {args.initial_model} loaded from previous state!")
        s = ECAPAModel(**vars(args))
        s.load_parameters(args.initial_model)
        epoch = 1
    elif len(modelfiles) >= 1:
        print(f"Model {modelfiles[-1]} loaded from previous state!")
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = ECAPAModel(**vars(args))
        s.load_parameters(modelfiles[-1])
    else:
        print("Hello, I called the model ... trainECAPAModel.py")
        epoch = 1
        s = ECAPAModel(**vars(args))
        print("Over calling model")

    print("Model loaded")

    EERs = []
    score_file = open(os.path.join(args.save_path, "score.txt"), "a+")

    while True:
        # Training for one epoch
        if epoch > 0:  # I should change it later if I want to train from the base
            print(f"Starting training for epoch {epoch}")
            loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)

        # Enrollment and Testing every [test_step] epochs
        if epoch % args.test_step == 0:
            print(f"Saving parameters for epoch {epoch}")
            s.save_parameters(args.save_path + "/model_%04d.model" % epoch)
            s.enroll_network(enroll_list=args.enroll_list, enroll_path=args.enroll_path, path_save_model=args.path_save_model)
            
            # Test with evaluation trials (without trial_type)
            # EER, minDCF, scores = s.test_network(test_list=args.eval_list, test_path=args.eval_path, path_save_model=args.path_save_model, compute_eer=False)
            
            # Test with development trials (with trial_type)
            print("Testing network")
            dev_EER, dev_minDCF, dev_scores = s.test_network(test_list=args.eval_list, test_path=args.eval_path, path_save_model=args.path_save_model, compute_eer=True)
            print("Hello I am here .... after test network ...")
            EERs.append(dev_EER)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, EER %2.2f%%, bestEER %2.2f%%" % (epoch, EERs[-1], min(EERs)))
            score_file.write("%d epoch, LR %f, LOSS %f, EER %2.2f%%, bestEER %2.2f%%\n" % (epoch, 1, loss, EERs[-1], min(EERs)))
            score_file.flush()

        if epoch >= args.max_epoch:
            break

        epoch += 1  # Increment epoch

    score_file.close()
    print("Training completed")

if __name__ == '__main__':
    main()
    print("I am in the main")