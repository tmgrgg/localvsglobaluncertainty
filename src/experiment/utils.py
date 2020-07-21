import os
from datetime import datetime
import pandas as pd


class ExperimentTable:

    def __init__(self, path, name, safe=False, include_time=True):
        """
        Class corresponding to a folder contained a single csv table alongside an images subfolder
        (TODO: handle images in write).
        :param path: path where experiment folder should persist
        :param name: name of experiment/experiment folder
        :param safe: assert that measure values (i.e. keys of dictionary's passed to write) all conform
                        to the same values as the first call to write.
        :param include_time: insert the time that the record was written as a measured value
        """
        self.path = path + '/' + name
        self.csv_path = self.path + '/table.csv'
        self._include_time = include_time

        if not os.path.isdir(self.path):
            print('::: ExperimentTable {} does not exist! :::'.format(self.path))
            self.__setup_dir__()
        else:
            self.__check_dir__()

        self._safe = safe
        if self._safe:
            self._columns = []

    def __setup_dir__(self):
        print('::: Building ExperimentTable :::')
        os.mkdir(self.path)
        os.mkdir(self.path + '/images')
        with open(self.path + "/readme.txt", "w") as txtfile:
            txtfile.write('Describe Experiment in terms of tabulated parameters and context.')

    def write(self, result_dict):
        if self._include_time:
            result_dict['_write_time'] = datetime.now()
        df = pd.DataFrame([result_dict])
        if os.path.exists(self.csv_path):
            if self._safe and set(result_dict.keys()) != self._columns:
                raise RuntimeError('Experiment Keys do not match! (Disable safe mode if this was intentional)')
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            print('Table empty: Creating {}'.format(self.csv_path))
            df.to_csv(self.csv_path, mode='w', header=True, index=False)
            if self._safe:
                self._columns = set(result_dict.keys())

    def __check_dir__(self):
        # TODO: implement safety checks on folder
        pass
