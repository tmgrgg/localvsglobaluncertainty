from PIL import Image
import os
import pandas as pd
import torch
from datetime import datetime


class ExperimentDirectory:

    def __init__(self, dir, name):
        self.path = os.path.join(dir, name)
        if not os.path.exists(self.path):
            print("Experiment directory does not exist, creating experiment directory {}.".format(self.path))
            os.mkdir(self.path)
            # add default subdirectories
            self.add_folder("models")
            self.add_folder("images")
        self.tables = {}
        for name in os.listdir(self.path):
            path = os.path.join(self.path, name)
            if os.path.isdir(path):
                self.__dict__.update({name + '_path': path})
            if os.path.isfile(path) and name.endswith('.csv'):
                self.add_table(name[:-4])

    def add_folder(self, name):
        path = os.path.join(self.path, name)
        if os.path.exists(path):
            return path
        print("Adding Folder {} to {}".format(name, self.path))
        os.mkdir(path)
        self.__dict__.update({name + '_path': path})
        return path

    def add_table(self, name):
        if name not in self.tables.keys():
            print("Adding Table {} to {}".format(name, self.path))
            table = ExperimentTable(exp_dir=self, name=name)
            self.tables.update({name: table})

    def save_state_dict(self, state_dict, name, folder='models'):
        folder_path = self.add_folder(folder)
        path = os.path.join(folder_path, '{}.pt'.format(name))
        print("Saving state dict {}".format(path))
        torch.save(state_dict, path)

    def save_image(self, image, name, folder='images'):
        folder_path = self.add_folder(folder)
        path = os.path.join(folder_path, '{}.png'.format(name))
        print("Saving image {}".format(path))
        image.save(path)

    def write(self, result_dict, table_name=None):
        table = self._infer_table(table_name)
        table.write(result_dict)

    def cached_state_dict(self, name, folder='models'):
        folder_path = self.add_folder(folder)
        path = os.path.join(folder_path, '{}.pt'.format(name))
        if os.path.isfile(path):
            print('Loading cached state dict for {} in {}'.format(name, folder_path))
            return torch.load(path), None
        return None, lambda state_dict: self.save_state_dict(state_dict, name=name, folder=folder)

    def cached_table_row(self, cache_dict, table_name=None):
        table = self._infer_table(table_name)
        if os.path.isfile(table.path):
            df = table.read()
            for key, val in cache_dict.items():
                df = df[df[key] == val]
            if len(df) != 0:
                print('Loading cached table row for {} in {}'.format(cache_dict, table.path))
                return df, None
        return None, lambda result_dict: self.write(result_dict.update(cache_dict) or result_dict,
                                                    table_name=table_name)

    def cached_image(self, name, folder='images'):
        folder_path = self.add_folder(folder)
        path = os.path.join(folder_path, '{}.png'.format(name))
        if os.path.isfile(path):
            print('Loading cached image {} in {}'.format(name, folder_path))
            return Image.open(path), None
        return None, lambda image: self.save_image(image, name=name, folder=folder)

    def cached_matplot_figure(self, name, folder='images'):
        folder_path = self.add_folder(folder)
        path = os.path.join(folder_path, '{}.png'.format(name))
        if os.path.isfile(path):
            print('Loading cached image {} in {}'.format(name, folder_path))
            return plt.imread(path), None
        return None, lambda fig: self.save_image(fig_to_image(fig), name=name, folder=folder)

    def _infer_table(self, table_name):
        if (len(self.tables) > 1) and (table_name is None):
            raise RuntimeError(
                "Cannot infer table: multiple tables in ExperimentDirectory. (Specify table argument from: {})"
                    .format(list(self.tables.keys())))
        elif table_name is None:
            table = list(self.tables.values())[0]
        else:
            try:
                table = self.tables[table_name]
            except KeyError as e:
                print('No such table {}'.format(table_name))
                raise e
        return table


class ExperimentTable:

    def __init__(self, exp_dir, name):
        """
        Class corresponding to a csv table
        :param exp_dir: ExperimentDirectory where table should be stored
        :param name: name of table
        """
        if not name.endswith('.csv'):
            name = name + '.csv'
        self.path = os.path.join(exp_dir.path, name)

    def write(self, result_dict, include_time=True, safe=True):
        if include_time:
            result_dict['_write_time'] = datetime.now()

        df = pd.DataFrame([result_dict])
        df = df.reindex(sorted(df.columns), axis=1)
        if os.path.exists(self.path):
            df.to_csv(self.path, mode='a', header=False, index=False)
        else:
            print('Creating table: {}'.format(self.path))
            df.to_csv(self.path, mode='w', header=True, index=False)

    def read(self):
        return pd.read_csv(self.path)


def fig_to_image(fig):
    fig.canvas.draw()
    return Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())


# Deprecated
# # For command line: all the setup could go in a setup() function, and then the command line need only parametrise
# # global variables (i.e. across all experiments/in this setup function). A general command line command for an instance
# # of CachedExperiment can then be used and we minimise bash writing.  Not sure this even makes sense.
# class CachedExperiment:
#
#     def __init__(self, table, run, caching_params=[]):
#         assert (type(caching_params) == list)
#         self._table = table
#         self._run = run
#         self._caching_params = caching_params
#
#     def run(self, *args, **kwargs):
#         # any params that are caching params must be passed to run as kwargs. \
#         if len(self._caching_params) > 0:
#             if os.path.exists(self._table.csv_path):
#                 df = pd.read_csv(self._table.csv_path)
#                 for key in self._caching_params:
#                     val = kwargs[key]
#                     df = df[df[key] == val]
#                 if len(df) != 0:
#                     print('Run already completed according to caching_params, skipping call to .run()!')
#                     return
#
#         result_dict = self._run(*args, **kwargs)
#         self._table.write(result_dict)
#
#     def __repr__(self):
#         return self._table.path


def track(tracker, res_train, res_valid, model='', plot=True):
    if res_train is not None:
        tracker.log(res_train['loss'], 'loss_' + model, setting='train')
        tracker.log(res_train['accuracy'], 'accuracy_' + model, setting='train')
    if res_valid is not None:
        tracker.log(res_valid['loss'], 'loss_' + model, setting='valid')
        tracker.log(res_valid['accuracy'], 'accuracy_' + model, setting='valid')
    if plot:
        tracker.plot()