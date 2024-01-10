import os
import torch
import numpy as np
import math
import sys
sys.path.append('./LTMU')
sys.path.append('./LTMU/DiMP_LTMU')
sys.path.append('./LTMU/DiMP_LTMU/pyMDNet/modules')
sys.path.append('./LTMU/DiMP_LTMU/pyMDNet/tracking')

# pymdnet
from pyMDNet.modules.model import *
from pyMDNet.modules.model import MDNet, BCELoss, set_optimizer
from pyMDNet.modules.sample_generator import SampleGenerator
from pyMDNet.tracking.run_tracker import forward_samples, train
from bbreg import BBRegressor
import yaml
opts = yaml.safe_load(open('./LTMU/DiMP_LTMU/pyMDNet/tracking/options.yaml','r'))

sys.path.append('./ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker as ByteTrack
from tools.track import make_parser

# hands-in-contact
from hic_detector import hic_config, hic_detect

from got10k.trackers import Tracker

class LTMUBT(Tracker):

    def __init__(self):
        super(LTMUBT, self).__init__(name='LTMU-BT', is_deterministic=False)

    def init(self, image, box):
        image = np.array(image)
   
        init_gt1 = box #[region.x, region.y, region.width, region.height]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]]  # ymin xmin ymax xmax

        # pyMDNet init
        self.t_id = 0
        self.last_gt = init_gt
        self.init_pymdnet(image, init_gt1)
        
        # HiC init
        self.hic, self.hic_cfg = hic_config()

        self.count = 0

        # ByteTrack init
        parser = make_parser()
        args = parser.parse_args([
            '-c', './ByteTrack/pretrained/bytetrack_nano_mot17.pth.tar'
        ])

        new_tracker = ByteTrack(args)
        self.new_tracker = new_tracker
        print("Byte Track initialized")

    def init_pymdnet(self, image, init_bbox):
        target_bbox = np.array(init_bbox)
        self.last_result = target_bbox
        self.pymodel = MDNet('./LTMU/DiMP_LTMU/pyMDNet/models/mdnet_imagenet_vid.pth')
        if opts['use_gpu']:
            self.pymodel = self.pymodel.cuda()
        self.pymodel.set_learnable_params(opts['ft_layers'])

        # Init criterion and optimizer
        self.criterion = BCELoss()
        init_optimizer = set_optimizer(self.pymodel, opts['lr_init'], opts['lr_mult'])
        self.update_optimizer = set_optimizer(self.pymodel, opts['lr_update'], opts['lr_mult'])

        # Draw pos/neg samples
        pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
            target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.feat_dim = pos_feats.size(-1)

        # Initial training
        train(self.pymodel, self.criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], opts=opts)
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                         opts['aspect_bbreg'])(
            target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
        bbreg_feats = forward_samples(self.pymodel, image, bbreg_examples, opts)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()
        # Init sample generators
        self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
        self.pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

        # Init pos/neg features for update
        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
        neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]

        self.calculate_score = lambda scores: sum(scores) / len(scores) if scores else 0

    def pymdnet_eval(self, image, samples):
        sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)
        return sample_scores[:, 1][:].cpu().numpy()

    def collect_samples_pymdnet(self, image):
        self.t_id += 1
        target_bbox = np.array([self.last_gt[1], self.last_gt[0], self.last_gt[3]-self.last_gt[1], self.last_gt[2]-self.last_gt[0]])
        pos_examples = self.pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
        if len(pos_examples) > 0:
            pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
            self.pos_feats_all.append(pos_feats)
        if len(self.pos_feats_all) > opts['n_frames_long']:
            del self.pos_feats_all[0]

        neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
        if len(neg_examples) > 0:
            neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
            self.neg_feats_all.append(neg_feats)
        if len(self.neg_feats_all) > opts['n_frames_short']:
            del self.neg_feats_all[0]

    def pymdnet_short_term_update(self):
        # Short term update
        nframes = min(opts['n_frames_short'], len(self.pos_feats_all))
        pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
        neg_data = torch.cat(self.neg_feats_all, 0)
        train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
              opts=opts)

    def pymdnet_long_term_update(self):
        if self.t_id % opts['long_interval'] == 0:
            # Long term update
            pos_data = torch.cat(self.pos_feats_all, 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            train(self.pymodel, self.criterion, self.update_optimizer, pos_data, neg_data, opts['maxiter_update'],
                  opts=opts)

    def update(self, image):
        image = np.array(image)

        candidate_bboxes = None

        x_min = int(self.last_gt[1]) - int(self.last_gt[3] / 2)
        y_min = int(self.last_gt[0]) - int(self.last_gt[2] / 2)
        x_max = x_min + int(self.last_gt[3]) - 1
        y_max = y_min + int(self.last_gt[2]) - 1

        output_results = torch.tensor([[x_min, y_min, x_max, y_max, 0.5]])

        img_h = image.shape[0]
        img_w = image.shape[1]

        scale_factor = 1.0
        img_h_out = int(image.shape[0] * scale_factor)
        img_w_out = int(image.shape[1] * scale_factor)

        new_tracker = self.new_tracker

        output_stracks = new_tracker.update(output_results, (img_h, img_w), (img_h_out, img_w_out))

        online_scores = []
        for stracks in output_stracks:
            online_scores.append(stracks.score)

        local_score = self.calculate_score(online_scores)

        pred_bbox = [int( self.last_gt[1]), int(self.last_gt[0]), int(self.last_gt[3] - self.last_gt[1]), int(self.last_gt[2] - self.last_gt[0])]
        local_state = np.array(pred_bbox)

        self.last_gt = np.array([local_state[1], local_state[0], local_state[1] + local_state[3], local_state[0] + local_state[2]])
        
        if local_score >= 0.5:
            flag = 'normal'
        else:
            flag = 'not_found'

        md_score = self.pymdnet_eval(image, np.array(local_state).reshape([-1, 4]))[0]
        self.score_max = md_score

        if md_score > 0 and flag == 'normal':
            self.flag = 'found'

            self.last_gt = np.array(
                [local_state[1], local_state[0], local_state[1] + local_state[3], local_state[0] + local_state[2]])
        elif md_score < 0 or flag == 'not_found':
            self.count += 1
            self.flag = 'not_found'

            candidate_bboxes = np.array(hic_detect(self.hic, self.hic_cfg, image))

            if candidate_bboxes.shape[0] > 0:
                candidate_scores = self.pymdnet_eval(image, candidate_bboxes)
            else:
                candidate_scores = np.zeros((1,1), dtype=np.float32)

            max_id = np.argmax(candidate_scores)
            if candidate_scores[max_id] > 0:
                redet_bboxes = candidate_bboxes[max_id]
                if self.count >= 5:
                    self.last_gt = np.array([redet_bboxes[1], redet_bboxes[0], redet_bboxes[1] + redet_bboxes[3],
                                             redet_bboxes[0] + redet_bboxes[2]])
                    self.score_max = candidate_scores[max_id]
                    self.count = 0

        if md_score > 0 and flag == 'normal':
            self.collect_samples_pymdnet(image)

        self.pymdnet_long_term_update()

        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]
 
        return [float(self.last_gt[1]), float(self.last_gt[0]), float(width), float(height)]
