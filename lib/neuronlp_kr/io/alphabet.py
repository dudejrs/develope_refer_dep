__author__ = 'Brian Kim'

import json
import pickle
import os
from neuronlp2.io.logger import get_logger
from neuronlp2.io.alphabet import Alphabet

class Alphabet(Alphabet):
    # def get_content(self):
    #     if self.singletons is None:
    #         return {'instance2index': self.instance2index, 'instances': self.instances}
    #     else:
    #         return {'instance2index': self.instance2index, 'instances': self.instances,
    #                 'singletions': list(self.singletons)}
    def get_content(self):
        if self.singletons is None:
            return {'instance2index': self.__stringfy(self.instance2index), 'instances': self.instances }
        else:
            return {'instance2index': self.__stringfy(self.instance2index), 'instances': self.instances ,
                    'singletions': list(self.singletons)}

    def __from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]
        if 'singletions' in data:
            self.singletons = set(data['singletions'])
        else:
            self.singletons = None

    def __stringfy(self,data) :
        data_ = {}
        for key in data.keys() :
            if type(key) is not str :
                try :
                    data_[str(key)] = data[key]
                except  :
                    try :
                        data_[repr(key)] = data[key]
                        pass
                    except Exception as e :
                        print(e)
            else :
                data_[key] = data[key]
        return data_

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            json.dump(self.get_content(),
                      open(os.path.join(output_directory, saving_name + ".json"), 'w'))
        except Exception as e:
            self.logger.warn("Alphabet is not saved: %s" % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.__from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
        self.next_index = len(self.instances) + self.offset
        self.keep_growing = False
