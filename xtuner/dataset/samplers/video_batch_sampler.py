from typing import Sequence

from torch.utils.data import BatchSampler, Sampler
from torch.utils.data import ConcatDataset as TorchConcatDataset
from mmengine.dist import get_dist_info
from mmengine.logging import print_log
from torch import distributed as dist

# TODO: maybe replace with a data_loader wrapper
class VideoImageSeperateBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __init__(
        self, sampler: Sampler, batch_size: int, video_batch_size=1, drop_last: bool = False
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, ' f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                'batch_size should be a positive integer value, ' f'but got batch_size={batch_size}'
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.video_batch_size = video_batch_size
        self.drop_last = drop_last
        # two groups for w < h and w >= h
        self._aspect_ratio_buckets = [[] for _ in range(3)]
        self.all_batches = self.compute_length()
        self.length = len(self.all_batches)
        # sychronize all batches of different processes
        rank, world_size = get_dist_info()
        print(f'rank {rank}, world_size {world_size}')
        if world_size > 1:
            gather_objects = [self.length for i in range(world_size)]  # any picklable object
            output = [None for _ in gather_objects]
            dist.all_gather_object(output, gather_objects[dist.get_rank()])
            length_max = max(output)
            if self.length < length_max:
                print_log(f'rank{rank} origin batchsize = {self.length}, padding to batchsize = {length_max}')
                repeat_num = length_max - self.length
                self.all_batches.extend([self.all_batches[-1] for i in range(repeat_num)])
                self.length = length_max
            assert len(self.all_batches) == length_max

    def __iter__(self) -> Sequence[int]:
        for batch_idx in self.all_batches:
            yield batch_idx
        # for idx in self.sampler:
        #     data_info = self.sampler.dataset.get_data_info(idx)
        #     modal = data_info
        #     if modal == 'text':
        #         bucket_id = 0
        #     elif modal == 'image':
        #         bucket_id = 1
        #     elif modal == 'video':
        #         bucket_id = 2
        #     else:
        #         print(f'modal = {modal}, error!')
        #         raise
        #     bucket = self._aspect_ratio_buckets[bucket_id]
        #     bucket.append(idx)
        #     # yield a batch of indices in the same aspect ratio group
        #     if bucket_id == 2 and len(bucket) == self.video_batch_size:
        #         print(modal)
        #         yield bucket[:]
        #         del bucket[:]
        #     elif bucket_id!= 2 and len(bucket) == self.batch_size:
        #         # continue
        #         print(modal)
        #         yield bucket[:]
        #         del bucket[:]

        # # yield the rest data and reset the bucket
        # left_data = self._aspect_ratio_buckets[0] + self._aspect_ratio_buckets[1]
        # left_video_data = self._aspect_ratio_buckets[2]
        # self._aspect_ratio_buckets = [[] for _ in range(3)]
        # while len(left_data) > 0:
        #     if len(left_data) <= self.batch_size:
        #         if not self.drop_last:
        #             yield left_data[:]
        #         left_data = []
        #     else:
        #         yield left_data[: self.batch_size]
        #         left_data = left_data[self.batch_size :]
        # while len(left_video_data) > 0:
        #     if len(left_video_data) <= self.video_batch_size:
        #         if not self.drop_last:
        #             yield left_video_data[:]
        #         left_video_data = []
        #     else:
        #         yield left_video_data[: self.video_batch_size]
        #         left_video_data = left_video_data[self.video_batch_size :]

    def __len__(self) -> int:
        # if self.drop_last:
        #     return len(self.sampler) // self.batch_size
        # else:
        #     return (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return self.length
    
    def find_sublist_index(self, lengths, f) -> int:
        # 计算总长度
        total_length = sum(lengths)
        # 如果 f 超出总长度范围，返回 None
        if f > total_length:
            return None
        # 计算偏移量
        offset = f - 1  # 注意索引从 0 开始，所以要减 1
        # 寻找偏移量所在的子列表
        for index, length in enumerate(lengths):
            if offset < length:
                return index
            offset -= length
        return None  # 如果未找到，返回 None

    def compute_length(self) -> int:
        buckets = [[] for _ in range(3)]
        all_batches = []
        video_batch = 0
        else_batch = 0
        if isinstance(self.sampler.dataset, TorchConcatDataset):
            dataset_length = []
            sub_dataset_list = []
            print('compute dataset length for each dataset')
            for sub_dataset in self.sampler.dataset.datasets:
                dataset_length.append(len(getattr(sub_dataset, 'modality_length')))
                sub_dataset_list.append(sub_dataset)
        for idx in self.sampler:
            if isinstance(self.sampler.dataset, TorchConcatDataset):
                dataset_index = self.find_sublist_index(dataset_length, idx)
                try:
                    dataset_inner_index = idx - sum(dataset_length[:dataset_index])
                    data_info = sub_dataset_list[dataset_index].get_data_info(dataset_inner_index)
                except Exception as e:
                    print(f"exception {e}")
                    print(idx, dataset_inner_index, dataset_index)
            else:
                data_info = self.sampler.dataset.get_data_info(idx)
            modal = data_info
            if modal == 'text':
                bucket_id = 0
            elif modal == 'image':
                bucket_id = 1
            elif modal == 'video':
                bucket_id = 2
            else:
                print(f'modal = {modal}, error!')
                raise
            bucket = buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if bucket_id == 2 and len(bucket) == self.video_batch_size:
                # print(modal)
                video_batch += 1
                all_batches.append(bucket[:])
                del bucket[:]
            elif bucket_id != 2 and len(bucket) == self.batch_size:
                # continue
                # print(modal)
                else_batch += 1
                all_batches.append(bucket[:])
                del bucket[:]
            # yield the rest data and reset the bucket

        left_data = buckets[0] + buckets[1]
        left_video_data = buckets[2]
        buckets = [[] for _ in range(3)]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    all_batches.append(left_data[:])
                    else_batch += 1
                left_data = []
            else:
                all_batches.append(left_data[: self.batch_size])
                else_batch += 1
                left_data = left_data[self.batch_size :]
        while len(left_video_data) > 0:
            if len(left_video_data) <= self.video_batch_size:
                if not self.drop_last:
                    all_batches.append(left_video_data[:])
                    video_batch += 1
                left_video_data = []
            else:
                all_batches.append(left_video_data[: self.video_batch_size])
                video_batch += 1
                left_video_data = left_video_data[self.video_batch_size :]
        print(f'video_batch number = {video_batch}, else_batch number = {else_batch}')
        return all_batches