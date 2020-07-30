import os
from datetime import datetime
import pandas as pd
from PIL.Image import Image
import torch


# what about if we had like an experiments class that just takes a bit of code and shoves it into the run method
# and has some predefined API with how it tracks and deals with/caches recorded quantities so that so long as the
# experiments class is instantiated and persisted in the same place it can be rerun without issue...
# would also possibly help to define experiments as command line tools in a more trivial way? Given that presumably
# with the way they did it before if there's a crash or you get booted off you have to rerun the experiments from scratch?
# actually that's not true... they did have that saving stuff. Perhaps that's enough?
# whta I've written here is for 'short repeated experiments'


class ExperimentTable:

    def __init__(self, path, name, safe=False, include_time=True):
        """
        Class corresponding to a folder contained a single csv table alongside an images subfolder
        (TODO: handle images in write).
        :param path: path where experiments folder should persist
        :param name: name of experiments/experiments folder
        :param safe: assert that measure values (i.e. keys of dictionary's passed to write) all conform
                        to the same values as the first call to write.
        :param include_time: insert the time that the record was written as a measured value
        """
        self.path = path + '/' + name
        self.csv_path = self.path + '/table.csv'
        self.image_path = self.path + '/images'
        self.model_path = self.path + '/models'
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
        os.mkdir(self.image_path)
        os.mkdir(self.model_path)
        with open(self.path + "/readme.txt", "w") as txtfile:
            txtfile.write('Describe Experiment in terms of tabulated parameters and context.')

    def write(self, result_dict):
        if self._include_time:
            result_dict['_write_time'] = datetime.now()

        result_dict = self._handle_images(result_dict)
        result_dict = self._handle_models(result_dict)

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

    def _handle_images(self, result_dict):
        for (key, val) in result_dict.items():
            if isinstance(val, Image):
                k = len(os.listdir(self.image_path))
                save_path = '{}/{}.png'.format(self.image_path, k)
                val.save(save_path)
                result_dict[key] = save_path
        return result_dict

    def _handle_models(self, result_dict):
        for (key, val) in result_dict.items():
            if isinstance(val, torch.nn.Module):
                print('Saving Model')
                k = len(os.listdir(self.model_path))
                save_path = '{}/{}.pt'.format(self.model_path, k)
                torch.save(val.state_dict(), save_path)
                result_dict[key] = save_path
        return result_dict


# For command line: all the setup could go in a setup() function, and then the command line need only parametrise
# global variables (i.e. across all experiments/in this setup function). A general command line command for an instance
# of CachedExperiment can then be used and we minimise bash writing.  Not sure this even makes sense.
class CachedExperiment:

    def __init__(self, table, run, caching_params=[]):
        assert (type(caching_params) == list)
        self._table = table
        self._run = run
        self._caching_params = caching_params

    def run(self, *args, **kwargs):
        # any params that are caching params must be passed to run as kwargs. \
        if os.path.exists(self._table.csv_path):
            df = pd.read_csv(self._table.csv_path)
            for key in self._caching_params:
                val = kwargs[key]
                df = df[df[key] == val]
            if len(df) != 0:
                print('Run already completed according to caching_params, skipping call to .run()!')
                return

        result_dict = self._run(*args, **kwargs)
        self._table.write(result_dict)


def track(tracker, res_train, res_valid, model='', plot=True):
    if res_train is not None:
        tracker.log(res_train['loss'], 'loss_' + model, setting='train')
        tracker.log(res_train['accuracy'], 'accuracy_' + model, setting='train')
    if res_valid is not None:
        tracker.log(res_valid['loss'], 'loss_' + model, setting='valid')
        tracker.log(res_valid['accuracy'], 'accuracy_' + model, setting='valid')
    if plot:
        tracker.plot()