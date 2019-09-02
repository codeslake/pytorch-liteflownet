#from utils import *
import collections
import numpy as np
import cv2
import os
from threading import Thread
from threading import Lock
from datetime import datetime
from pathlib import Path
import torch

class Data_Loader:
    def __init__(self, config, is_train, name, thread_num = 3):

        self.name = name
        self.config = config
        self.is_train = is_train
        self.thread_num = thread_num
        self.sample_num = config.sample_num

        self.num_partition = 1
        self.skip_length = np.array(self.config.skip_length)
        self.skip_length_reverse = np.array(self.config.skip_length_reverse)

        self.blur_folder_path_list, self.blur_file_path_list, _ = self._load_file_list(self.config.frame_offset, self.config.blur_path)
        self.sharp_folder_path_list, self.sharp_file_path_list, _ = self._load_file_list(self.config.frame_offset, self.config.sharp_path)
        self.of_folder_path_list, self.of_frame_path_list, _ = self._load_file_list(config.of_path)
        self.of_folder_reverse_path_list, self.of_frame_reverse_path_list, _ = self._load_file_list(config.of_reverse_path)

        self.h = self.config.height
        self.w = self.config.width
        self.batch_size = self.config.batch_size
        self.is_color = True

        self.is_augment = self.config.is_augment
        self.is_reverse = self.config.is_reverse

    def init_data_loader(self, inputs):

        self.idx_video = []
        self.idx_frame = []
        self.init_idx()

        self.num_itr = int(np.ceil(len(sum(self.idx_frame, [])) / self.batch_size))

        self.lock = Lock()
        self.is_end = False

        ### THREAD HOLDERS ###
        self.net_placeholder_names = None

        self.net_placeholder_names = list(inputs.keys())
        self.net_inputs = inputs

        self.threads = [None] * self.thread_num
        self.threads_unused = [None] * self.thread_num
        self.feed_dict_holder = self._set_feed_dict_holder(self.net_placeholder_names, self.thread_num)
        self._init_data_thread()

        # self._print_log()

    def init_idx(self):
        self.idx_video = []
        self.idx_frame = []
        for i in range(len(self.sharp_file_path_list)):
            total_frames = len(self.sharp_file_path_list[i])

            idx_frame_temp = list(range(0, total_frames - (self.skip_length[-1] - self.skip_length[0]) ))

            self.idx_frame.append(idx_frame_temp)
            self.idx_video.append(i)

        self.is_end = False

    def get_batch(self, threads_unused, thread_idx):
        assert(self.net_placeholder_names is not None)
        # tl.logging.debug('[%s] \tthread[%s] > get_batch start [%s]' % (self.name, str(thread_idx), str(datetime.now())))

        ## random sample frame indexes
        self.lock.acquire()
        # tl.logging.debug('[%s] \t\tthread[%s] > acquired lock [%s]' % (self.name, str(thread_idx), str(datetime.now())))

        if self.is_end:
            # tl.logging.debug('[%s] \t\tthread[%s] > releasing lock 1 [%s]' % (self.name, str(thread_idx), str(datetime.now())))
            self.lock.release()
            return

        video_idxes = []
        frame_offsets = []

        actual_batch = 0
        for i in range(0, self.batch_size):
            if i == 0 and len(self.idx_video) == 0:
                self.is_end = True
                # tl.logging.debug('[%s] \t\tthread[%s] > releasing lock 2 [%s]' % (self.name, str(thread_idx), str(datetime.now())))
                self.lock.release()
                return
            # original
            # elif i > 0 and len(self.idx_video) == 0:
            #     break
            # ignore actual batch < batch
            elif i > 0 and len(self.idx_video) == 0:
                self.is_end = True
                # tl.logging.debug('[%s] \t\tthread[%s] > releasing lock 2 [%s]' % (self.name, str(thread_idx), str(datetime.now())))
                self.lock.release()
                return

            else:
                if self.is_train:
                    idx_x = np.random.randint(len(self.idx_video))
                    video_idx = self.idx_video[idx_x]
                    idx_y = np.random.randint(len(self.idx_frame[video_idx]))
                else:
                    idx_x = 0
                    idx_y = 0
                    video_idx = self.idx_video[idx_x]

            frame_offset = self.idx_frame[video_idx][idx_y]
            video_idxes.append(video_idx)
            frame_offsets.append(frame_offset)
            self._update_idx(idx_x, idx_y)
            actual_batch += 1

        # tl.logging.debug('[%s] \t\tthread[%s] > releasing lock 4 [%s]' % (self.name, str(thread_idx), str(datetime.now())))
        self.lock.release()
        threads_unused[thread_idx] = True

        ## init thread lists
        data_holder = self._set_data_holder(self.net_placeholder_names, actual_batch)

        ## start thread
        threads = [None] * actual_batch
        for batch_idx in range(actual_batch):
            video_idx = video_idxes[batch_idx]
            frame_offset = frame_offsets[batch_idx]
            is_augment = np.random.randint(2) if self.is_augment else 0 # 0: none, 1: flip horizontal
            is_reverse = np.random.randint(2) if self.is_reverse else 0
            threads[batch_idx] = Thread(target = self.read_dataset, args = (data_holder, batch_idx, video_idx, frame_offset, is_augment, is_reverse))
            threads[batch_idx].start()

        for batch_idx in range(actual_batch):
            threads[batch_idx].join()

        for (key, val) in data_holder.items():
            data_holder[key] = np.concatenate(data_holder[key][0 : actual_batch], axis = 0)

        for holder_name in self.net_placeholder_names:
            self.feed_dict_holder[holder_name][thread_idx] = torch.FloatTensor(data_holder[holder_name].transpose(0, 3, 1, 2)).cuda()

        #tl.logging.debug('[%s] \tthread[%s] > get_batch done [%s]' % (self.name, str(thread_idx), str(datetime.now())))

    def read_dataset(self, data_holder, batch_idx, video_idx, frame_offset, is_augment, is_reverse):
        #sampled_frame_idx = np.arange(frame_offset, frame_offset + self.sample_num * self.skip_length, self.skip_length)
        if is_reverse:
            sampled_frame_idx = frame_offset + self.skip_length_reverse
        else:
            sampled_frame_idx = frame_offset + self.skip_length

        s_temp = [None] * len(sampled_frame_idx)
        b_temp = [None] * len(sampled_frame_idx)

        threads = [None] * len(sampled_frame_idx)
        for frame_idx in range(len(sampled_frame_idx)):
            is_prev = False if frame_idx == len(sampled_frame_idx) - 1 else True
            is_read_of = True if frame_idx == len(sampled_frame_idx) - 2 else False

            sampled_idx = sampled_frame_idx[frame_idx]

            threads[frame_idx] = Thread(target = self.read_frame_data, args = (data_holder, batch_idx, video_idx, frame_idx, sampled_idx, b_temp, s_temp, is_prev, is_augment, is_reverse))
            threads[frame_idx].start()

        for frame_idx in range(len(sampled_frame_idx)):
            threads[frame_idx].join()

 
        cropped_frames = np.concatenate([data_holder['b_t_1'][batch_idx], data_holder['b_t'][batch_idx], data_holder['s_t_1'][batch_idx], data_holder['s_t'][batch_idx]], axis = 3)

        if self.name == 'train':
            cropped_frames = self.crop_multi(cropped_frames, self.h, self.w, is_random = True)
        else:
            cropped_frames = self.crop_multi(cropped_frames, self.h, self.w, is_random = False)

        data_holder['b_t_1'][batch_idx] = cropped_frames[:, :, :, 0:3]
        data_holder['b_t'][batch_idx] = cropped_frames[:, :, :, 3:6]
        data_holder['s_t_1'][batch_idx] = cropped_frames[:, :, :, 6:9]
        data_holder['s_t'][batch_idx] = cropped_frames[:, :, :, 9:12]


    def read_frame_data(self, data_holder, batch_idx, video_idx, frame_idx, sampled_idx, b_temp, s_temp, is_prev, is_augment, is_reverse):
        # read stab frame
        blur_file_path = self.blur_file_path_list[video_idx]
        sharp_file_path = self.sharp_file_path_list[video_idx]

        assert(self._get_folder_name(str(Path(blur_file_path[sampled_idx]).parent)) == self._get_folder_name(str(Path(sharp_file_path[sampled_idx]).parent)))
        assert(self._get_base_name(blur_file_path[sampled_idx]) == self._get_base_name(sharp_file_path[sampled_idx]))

        b_frame = self._read_frame(blur_file_path[sampled_idx], is_augment, is_color = self.is_color)
        s_frame = self._read_frame(sharp_file_path[sampled_idx], is_augment, is_color = self.is_color)

        if is_prev:
            data_holder['b_t_1'][batch_idx] = b_frame
            data_holder['s_t_1'][batch_idx] = s_frame
        else:
            data_holder['b_t'][batch_idx] = b_frame
            data_holder['s_t'][batch_idx] = s_frame


    def _update_idx(self, idx_x, idx_y):
        video_idx = self.idx_video[idx_x]
        del(self.idx_frame[video_idx][idx_y])

        if len(self.idx_frame[video_idx]) == 0:
            del(self.idx_video[idx_x])
            # if len(self.idx_video) != 0:
            #     self.video_name = os.path.basename(self.sharp_file_path_list[self.idx_video[0]])

    def _load_file_list(self, root_path, child_path = None):
        folder_paths = []
        filenames_pure = []
        filenames_structured = []
        num_files = 0
        for root, dirnames, filenames in os.walk(root_path):
            if len(dirnames) == 0:
                if root[0] == '.':
                    continue
                if child_path is not None and child_path not in root: 
                    continue
                folder_paths.append(root)
                filenames_pure = []
                for i in np.arange(len(filenames)):
                    if filenames[i][0] != '.':
                        filenames_pure.append(os.path.join(root, filenames[i]))
                filenames_structured.append(np.array(sorted(filenames_pure)))
                num_files += len(filenames_pure)

        folder_paths = np.array(folder_paths)
        filenames_structured = np.array(filenames_structured)

        sort_idx = np.argsort(folder_paths)
        folder_paths = folder_paths[sort_idx]
        filenames_structured = filenames_structured[sort_idx]

        return np.squeeze(folder_paths), np.squeeze(filenames_structured), np.squeeze(num_files)

    def _read_frame(self, path, is_augment, is_color):
        if is_color:
            frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
            #frame = cv2.resize(frame, (self.w, self.h))
        else:
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame / 255.
            #frame = cv2.resize(frame, (self.w, self.h))

        if is_augment == 1: # flip vertical
            frame = cv2.flip(frame, 1)

        if is_color:
            return np.expand_dims(frame, axis = 0)
        else:
            return np.expand_dims(np.expand_dims(frame, axis = 2), axis = 0)

    def _read_surf(self, file_path, is_augment):
        rawdata = np.loadtxt(file_path)
        s_w = 1280
        s_h = 720

        if len(rawdata.shape) == 2:
            output = np.zeros((2,rawdata.shape[0],2))
            for i in range(rawdata.shape[0]):
                output[0,i,0] = int(int(np.round(rawdata[i,0])) / s_w * self.w) - 1
                output[0,i,1] = int(int(np.round(rawdata[i,1])) / s_h * self.h) - 1
                output[1,i,0] = int(int(np.round(rawdata[i,2])) / s_w * self.w) - 1
                output[1,i,1] = int(int(np.round(rawdata[i,3])) / s_h * self.h) - 1

            if is_augment == 1:
                output[:, :, 0] = (self.w - 1) - output[:, :, 0]

            return np.expand_dims(output, axis = 0)
        else:
            return np.zeros((1, 2, 0, 2))

    def _read_of(self, file_path, is_augment, h, w):
        of_t = cv2.resize(np.load(file_path), (w, h)) * 2

        of_t_x = of_t[:, :, 0]
        of_t_y = of_t[:, :, 1]

        if is_augment == 1:
            of_t_x = cv2.flip(of_t_x, 1)
            of_t_y = cv2.flip(of_t_y, 1)

        of_t_x = np.expand_dims(of_t_x, axis = 2)
        of_t_y = np.expand_dims(of_t_y, axis = 2)

        of_t = np.expand_dims(np.concatenate((of_t_x, of_t_y), axis = 2), axis = 0)

        return of_t

    def _get_base_name(self, path):
        return os.path.basename(path.split('.')[0])

    def _get_folder_name(self, path):
        path = os.path.dirname(path)
        return path.split(os.sep)[-1]

    def _set_feed_dict_holder(self, holder_names, thread_num):
        feed_dict_holder = collections.OrderedDict()
        for holder_name in holder_names:
            feed_dict_holder[holder_name] = [None] * thread_num

        return feed_dict_holder

    def _set_data_holder(self, net_placeholder_names, batch_num):
        data_holder = collections.OrderedDict()
        for holder_name in net_placeholder_names:
            data_holder[holder_name] = [None] * batch_num

        return data_holder

    def _init_data_thread(self):
        self.init_idx()
        #tl.logging.debug('[%s] INIT_THREAD [%s]' % str(self.name, datetime.now()))
        for thread_idx in range(0, self.thread_num):
            self.threads[thread_idx] = Thread(target = self.get_batch, args = (self.threads_unused, thread_idx))
            self.threads_unused[thread_idx] = False
            self.threads[thread_idx].start()

        #tl.logging.debug('[%s] INIT_THREAD DONE [%s]' % str(self.name, datetime.now()))

    def get_feed(self):
        thread_idx, is_end = self._get_thread_idx()
        #tl.logging.debug('[%s] THREAD[%s] > FEED_THE_NETWORK [%s]' % (self.name, str(thread_idx), str(datetime.now())))
        if is_end:
            return None, is_end

        # feed_dict = collections.OrderedDict()
        for (key, val) in self.net_inputs.items():
            self.net_inputs[key] = self.feed_dict_holder[key][thread_idx]

        #tl.logging.debug('[%s] THREAD[%s] > FEED_THE_NETWORK DONE [%s]' % (self.name, str(thread_idx), str(datetime.now())))
        return self.net_inputs, is_end

    def _get_thread_idx(self):
        for thread_idx in np.arange(self.thread_num):
            if self.threads[thread_idx].is_alive() == False and self.threads_unused[thread_idx] == False:
                    self.threads[thread_idx] = Thread(target = self.get_batch, args = (self.threads_unused, thread_idx))
                    self.threads[thread_idx].start()

        while True:
            is_unused_left = False
            for thread_idx in np.arange(self.thread_num):
                if self.threads_unused[thread_idx]:
                    is_unused_left = True
                    if self.threads[thread_idx].is_alive() == False:
                        self.threads_unused[thread_idx] = False
                        return thread_idx, False

            if is_unused_left == False and self.is_end:
                self._init_data_thread()
                return None, True

    def _print_log(self):
        print('sharp_folder_path_list')
        print(len(self.sharp_folder_path_list))

        print('sharp_file_path_list')
        total_file_num = 0
        for file_path in self.sharp_file_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('blur_file_path_list')
        total_file_num = 0
        for file_path in self.blur_file_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('of_file_path_list')
        total_file_num = 0
        for file_path in self.of_frame_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('of_reverse_file_path_list')
        total_file_num = 0
        for file_path in self.of_frame_reverse_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('num itr per epoch')
        print(self.num_itr)

    def crop_multi(self, x, wrg, hrg, is_random=False, row_index=0, col_index=1):
        """Randomly or centrally crop multiple images.

        Parameters
        ----------
        x : list of numpy.array
            List of images with dimension of [n_images, row, col, channel] (default).
        others : args
            See ``tl.prepro.crop``.

        Returns
        -------
        numpy.array
            A list of processed images.

        """
        h, w = x[0].shape[row_index], x[0].shape[col_index]

        if (h <= hrg) or (w <= wrg):
            raise AssertionError("The size of cropping should smaller than the original image")

        if is_random:
            h_offset = int(np.random.uniform(0, h - hrg) - 1)
            w_offset = int(np.random.uniform(0, w - wrg) - 1)
            results = []
            for data in x:
                results.append(data[int(h_offset):int(hrg + h_offset), int(w_offset):int(wrg + w_offset)])
            return np.asarray(results)
        else:
            # central crop
            h_offset = (h - hrg) / 2
            w_offset = (w - wrg) / 2
            results = []
            for data in x:
                results.append(data[int(h_offset):int(h - h_offset), int(w_offset):int(w - w_offset)])
            return np.asarray(results)
