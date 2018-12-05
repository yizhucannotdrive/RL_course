import json
import numpy as np
from s2clientprotocol import ui_pb2 as sc_ui
from s2clientprotocol import spatial_pb2 as sc_spatial


class Config:
    # TODO extract embed_dim_fn to config
    def __init__(self, sz, map, run_id, embed_dim_fn=lambda x: max(1, round(np.log2(x)))):
        self.run_id = run_id
        self.sz, self.map = sz, map
        self.embed_dim_fn = embed_dim_fn
        self.feats = self.acts = self.act_args = self.arg_idx = self.ns_idx = None

    def build(self, cfg_path):
        feats, acts, act_args = self._load(cfg_path)
        if 'screen' not in feats:
            feats['screen'] = features.SCREEN_FEATURES._fields
        if 'minimap' not in feats:
            feats['minimap'] = features.MINIMAP_FEATURES._fields
        if 'non_spatial' not in feats:
            feats['non_spatial'] = NON_SPATIAL_FEATURES.keys()
        self.feats = feats
        # TODO not connected to anything atm
        if acts is None:
            acts = FUNCTIONS
        self.acts = acts

        if act_args is None:
            act_args = TYPES._fields
        self.act_args = act_args

        self.arg_idx = {arg: i for i, arg in enumerate(self.act_args)}
        self.ns_idx = {f: i for i, f in enumerate(self.feats['non_spatial'])}
        #print(self.feats)
        #print(self.acts)
        #print(self.act_args)
        #print(self.arg_idx)
        #print(self.ns_idx)

    def map_id(self):
        return self.map + str(self.sz)

    def full_id(self):
        if self.run_id == -1:
            return self.map_id()
        return self.map_id() + "/" + str(self.run_id)

    def policy_dims(self):
        return [(len(self.acts), 0)] + [(getattr(TYPES, arg).sizes[0], is_spatial(arg)) for arg in self.act_args]

    def screen_dims(self):
        return self._dims('screen')

    def minimap_dims(self):
        return self._dims('minimap')

    def non_spatial_dims(self):
        return [NON_SPATIAL_FEATURES[f] for f in self.feats['non_spatial']]

    # TODO maybe move preprocessing code into separate class?
    def preprocess(self, obs):
        #types ['screen', 'minimap', 'units', "player", "available_actions"]
        #return [self._preprocess(obs, _type) for _type in ['screen', 'minimap', 'units'] + self.feats['non_spatial']]
        return [self._preprocess(obs, _type) for _type in ['units'] + self.feats['non_spatial']]

    def _dims(self, _type):
        return [f.scale**(f.type == CAT) for f in self._feats(_type)]

    def _feats(self, _type):
        feats = getattr(features, _type.upper() + '_FEATURES')
        return [getattr(feats, f_name) for f_name in self.feats[_type]]

    def _preprocess(self, obs, _type):
        if _type in self.feats['non_spatial']:
            return np.array([self._preprocess_non_spatial(ob, _type) for ob in obs])
        #_type could be "minimap" or "screen"
        if _type == "units":
            # we can choose return more, "float","int", "bool"
            out = []
            units_all_info = []
            for ob in obs:
                units_info =[]
                good_units = []
                bad_units = []
                for i in range(len(ob[_type])):
                    units_info+=list(ob[_type][i]["float"])
                    if ob[_type][i]["int"] and ob[_type][i]["int"].alliance == 1:
                        #ob[_type][i]["float"] is a numpy class which we can call arribute by names, pos_x etc
                        good_units.append(ob[_type][i]["float"])
                    if ob[_type][i]["int"] and ob[_type][i]["int"].alliance !=1:
                        bad_units.append(ob[_type][i]["float"])
                #hack here, 10 means defendroaches, we have 10 units; 4 means we have 4 enemy units
                while len(good_units)<10:
                    good_units.append(np.zeros(13))
                while len(bad_units)<10:
                    bad_units.append(np.zeros(13))
                out.append([good_units, bad_units])
                units_all_info.append(units_info)
            #print(len(units_all_info[0])) #169
            #exit()
            return np.array(units_all_info)
            #return np.array(out)
        spatial = [[ob[_type][f.index] for f in self._feats(_type)] for ob in obs]
        return np.array(spatial).transpose((0, 2, 3, 1))

    def _preprocess_non_spatial(self, ob, _type):
        if _type == 'available_actions':
            acts = np.zeros(len(self.acts))
            acts[ob['available_actions']] = 1
            return acts
        return ob[_type]

    def save(self, cfg_path):
        with open(cfg_path, 'w') as fl:
            json.dump({'feats': self.feats, 'act_args': self.act_args}, fl)

    def _load(self, cfg_path):
        with open(cfg_path, 'r') as fl:
            data = json.load(fl)
        return data.get('feats'), data.get('acts'), data.get('act_args')


def is_spatial(arg):
    return arg in ['screen', 'screen2', 'minimap']
