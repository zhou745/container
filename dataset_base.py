import numpy as np
from torch.utils.data import Dataset
from os.path import expanduser, abspath

class BaseDataset(Dataset):
    def __init__(self,
                 conf_path = "/mnt/cache/zhoujingqiu.vendor/universal_pix2seq",
                 read_from = "petrel"):

        super(BaseDataset, self).__init__()

        self.conf_path = abspath(expanduser(conf_path))
        self.read_from = read_from

        self.count = 0

        if read_from=="petrel":
            try:
                from petrel_client.client import Client
            except:
                raise RuntimeError("Can not find petrel lib")

            config_path_file = conf_path+"/.petreloss.conf"
            print(config_path_file,flush=True)
            self.client = Client(enable_mc=True, conf_path=config_path_file)
        elif read_from=="mc":
            try:
                import mc
            except:
                raise RuntimeError("Can not find mc lib")
            server_list_config_file = conf_path+"/.server_list.conf"
            client_config_file = conf_path+"/.client.conf"
            self.client = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
        elif read_from=="ori":
            self.client = None
        else:
            raise RuntimeError("unknown loading type")

        self.average = [0. for i in range(4)]
        self.alpha = 0.01

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


    def read_file(self, filename):
        if self.read_from == 'mc':
            try:
                import mc
            except:
                raise RuntimeError("can not find mc")

            value = mc.pyvector()
            self.client.Get(filename, value)
            value_str = mc.ConvertBuffer(value)
            filebytes = np.frombuffer(value_str.tobytes(), dtype=np.uint8)

        elif self.read_from == 'petrel':
            value = self.client.Get(filename)
            filebytes = np.frombuffer(value, dtype=np.uint8)
            # self.count += 1
            # print("read file all number %d"%(self.count),flush=True)
        else:
            raise RuntimeError("unknown value for read_from: {}".format(self.read_from))

        return filebytes

    def dump(self, writer, output):

        raise NotImplementedError
