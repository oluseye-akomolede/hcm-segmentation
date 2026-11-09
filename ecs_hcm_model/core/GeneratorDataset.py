import os
import pyarrow
import torch
from s3fs import S3FileSystem
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import fsspec

class GeneratorDataset(IterableDataset):
    def arrow_file_generator(self,shards: list , s3 : S3FileSystem):
        fs_from_s3 = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3))
        for file_path in shards:
            print(f"Reading file: {file_path}")
            try:
                
                with fs_from_s3.open_input_stream(file_path) as stream:
                    reader = pyarrow.ipc.open_stream(stream)
                    for batch in reader:
                        #print(batch.to_pydict())
                        yield batch.to_pydict()
                
            except pyarrow.lib.ArrowInvalid as e:
                print(f"Error reading file {file_path}: {e}")
                continue



    

    def __init__(self, rank=0, world_size=1):
        super(GeneratorDataset, self).__init__()
        self.access_key = os.environ["MINIO_ACCESS_KEY"]
        self.secret_key = os.environ["MINIO_SECRET_KEY"]
        self.url = "http://minio.minio.svc.cluster.local:9000"
        self.s3_storage_options = {}
        self.split = ""
        self.test_fs : S3FileSystem
        self.test_fs = None

        self.arrow_files = []
        self.generator = None
        self.transform = None

        self.rank = rank
        self.world_size = world_size
        
    def worker_initializer(self, transform : callable, split : str):
        self.s3_storage_options = {

                "key" : self.access_key,
                "secret" : self.secret_key,

                "endpoint_url" : self.url,
                "client_kwargs" : {"verify" : False},
            }
        self.split = split
        dataset_path = ""
        if self.split == 'train':
            dataset_path="hugging-face-bucket-imagenet-1k-train"
        elif self.split == "test":
            dataset_path="hugging-face-bucket-imagenet-1k-test"
        else:
            dataset_path="hugging-face-bucket-imagenet-1k-validation"
        S3_URI = f"s3://{dataset_path}"
        self.test_fs : S3FileSystem
        self.test_fs, path = fsspec.url_to_fs(S3_URI,**self.s3_storage_options)
        all_files = self.test_fs.ls(S3_URI)
        self.arrow_files = [file for file in all_files if file.endswith('.arrow')]
        self.transform = transform

    def __iter__(self):
        worker = get_worker_info()
        # A rank's stream must be split across its DataLoader workers as well as
        # across ranks, or every worker replays the whole stream. Outside a worker
        # process get_worker_info() is None and the split is across ranks only.
        worker_id, num_workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        shards = self.world_size * num_workers
        index = self.rank * num_workers + worker_id
        generator = self.arrow_file_generator(shards=self.arrow_files, s3=self.test_fs)
        if self.transform:
          for i, item in enumerate(generator):
            if i % shards == index:
                yield self.transform(item)
        else:
          for i, item in enumerate(generator):
            if i % shards == index:
                yield item
