import math
import torch
import torch.nn as nn
from .utils import init_tensor
from .contractables import MatRegion, OutputCore, ContractableList, \
                          EdgeVec
#3
class MPS(nn.Module):

    def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2,
                 label_site=None, init_std=1e-2, use_bias=True,
                 fixed_bias=True, use_GPU=False,parallel=False):
        super().__init__()
        #print("MPS __init__")

        if label_site is None:
            label_site = input_dim // 2
        assert label_site >= 0 and label_site <= input_dim


        module_list = []
        '''init_args = {'bond_str': 'slri',
                     'shape': [label_site, bond_dim, bond_dim, feature_dim],
                     'init_method': ('random_zero', init_std, output_dim)}'''
        init_args = {'bond_str': 'slri',
                     'shape': [label_site, bond_dim, bond_dim, feature_dim],
                     'init_std': init_std}

        if label_site > 0:
            tensor = init_tensor(**init_args)
            

            module_list.append(InputRegion(tensor, use_bias=use_bias, 
                                           fixed_bias=fixed_bias,use_GPU=use_GPU,parallel=parallel))
            #print('here is inputregion one')


        tensor = init_tensor(shape=[output_dim, bond_dim, bond_dim],
            bond_str='olr', init_std = init_std)
        module_list.append(OutputSite(tensor))
        #print('here is outputsite')

        if label_site < input_dim:
            init_args['shape'] = [input_dim-label_site, bond_dim, bond_dim, 
                                  feature_dim]
            tensor = init_tensor(**init_args)
            module_list.append(InputRegion(tensor, use_bias=use_bias, 
                                           fixed_bias=fixed_bias,use_GPU=use_GPU,parallel=parallel))
            #print('here is input region')
        self.linear_region = LinearRegion(module_list=module_list,parallel=parallel)
        assert len(self.linear_region) == input_dim
        

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.feature_dim = feature_dim
        self.label_site = label_site
        self.use_bias = use_bias
        self.fixed_bias = fixed_bias
        #self.cutoff = cutoff
        #self.merge_threshold = merge_threshold
        self.feature_map = None
        self.use_GPU = use_GPU
        self.parallel = parallel

    def forward(self, input_data):
        #print("MPS forward")

        input_data = self.embed_input(input_data)
        output = self.linear_region(input_data)
        #print('output.device : ',output.device)

        if isinstance(output, tuple):
            output, new_bonds, new_svs = output

            assert len(new_bonds) == len(self.bond_list)
            assert len(new_bonds) == len(new_svs)
            for i, bond_dim in enumerate(new_bonds):
                if bond_dim != -1:
                    assert new_svs[i] is not -1
                    self.bond_list[i] = bond_dim
                    self.sv_list[i] = new_svs[i]

        return output

    def embed_input(self, input_data):
        #print("MPS embed_input")

        assert len(input_data.shape) in [2, 3]
        assert input_data.size(1) == self.input_dim

        if len(input_data.shape) == 3:
            if input_data.size(2) != self.feature_dim:
                #raise ValueError(f"input_data has wrong shape to be unembedded "
                #"or pre-embedded data (input_data.shape = "
                #f"{list(input_data.shape)}, feature_dim = {self.feature_dim})")
                raise ValueError("input_data has wrong shape to be unembedded "
                "or pre-embedded data (input_data.shape = "
                "{}, feature_dim = {})".format(list(input_data.shape),self.feature_dim))
            return input_data

        embedded_shape = list(input_data.shape) + [self.feature_dim]
        

        if self.feature_map is not None:
            '''
            2020/4/27
            Do not use for loop !
            Mapping tensor component by component.then stack them all togethor by dim = -1 

            '''
            f_map = self.feature_map
            #embedded_data = torch.stack([torch.stack([f_map(x) for x in batch])
            #                                          for batch in input_data])
            #embedded_data = torch.stack([input_data,1-input_data],dim = -1)
            embedded_data = torch.stack(f_map(input_data),dim = -1)
            


            assert embedded_data.shape == torch.Size([input_data.size(0), self.input_dim, self.feature_dim])


        else:
            if self.feature_dim != 2:
                #raise RuntimeError(f"self.feature_dim = {self.feature_dim}, "
                #     "but default feature_map requires self.feature_dim = 2")
                raise RuntimeError("self.feature_dim = {}, "
                      "but default feature_map requires self.feature_dim = 2".format(self.feature_dim))

            embedded_data = torch.stack([input_data, 1 - input_data], dim=2)

        return embedded_data

    def register_feature_map(self, feature_map):
        #print("MPS register_feature_map")

        if feature_map is not None:
            out_shape = torch.stack(feature_map(torch.tensor(0)),dim=-1).shape
            #out_shape = feature_map(torch.tensor(0)).shape
            needed_shape = torch.Size([self.feature_dim])
            if out_shape != needed_shape:
                #raise ValueError("Given feature_map returns values of size "
                     #           f"{list(out_shape)}, but should return "
                      #          f"values of size {list(needed_shape)}")
                raise ValueError("Given feature_map returns values of size "
                                "{}, but should return "
                                "values of size {}".format(list(out_shape),list(needed_shape)))

        self.feature_map = feature_map

    def core_len(self):
        #print("MPS core_len")

        return self.linear_region.core_len()

    def __len__(self):
        #print("MPS __len__")

        return self.input_dim


class LinearRegion(nn.Module):

    def __init__(self, module_list,module_states=None,parallel=False):
        #print("LinearRegion __init__")

        if not isinstance(module_list, list) or module_list is []:
            raise ValueError("Input to LinearRegion must be nonempty list")
        for i, item in enumerate(module_list):
            if not isinstance(item, nn.Module):
                #raise ValueError("Input items to LinearRegion must be PyTorch "
                          #      f"Module instances, but item {i} is not")
                raise ValueError("Input items to LinearRegion must be PyTorch "
                                "Module instances, but item {} is not".format(i))
        super().__init__()


        self.module_list = nn.ModuleList(module_list)
        self.parallel = parallel

    def forward(self, input_data):
        #print("LinearRegion forward")

        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        lin_bonds = ['l', 'r']

        #to_cuda = input_data.is_cuda
        #input_data.cuda()
        #device = f'cuda:{input_data.get_device()}' if to_cuda else 'cpu'

        ind = 0
        contractable_list = []
        for module in self.module_list:
            mod_len = len(module)
            if mod_len == 1:
                mod_input = input_data[:, ind]
            else:
                mod_input = input_data[:, ind:(ind+mod_len)]
            ind += mod_len
            
            #print("### entry to contractable_list ###")
            contractable_list.append(module(mod_input))
            #print("### exit contractable_list ###")




        end_items = [contractable_list[i]for i in [0, -1]]
        bond_strs = [item.bond_str for item in end_items]
        bond_inds = [bs.index(c) for (bs, c) in zip(bond_strs, lin_bonds)]
        bond_dims = [item.tensor.size(ind) for (item, ind) in
                                               zip(end_items, bond_inds)]


        end_vecs = [torch.zeros(dim) for dim in bond_dims]
        #print("### start to replace end_vecs[0] to 1 ###")
        for vec in end_vecs:
            vec[0] = 1
        #print("### done replace end_vecs[0] to 1 ###")

        #print("### append EdgeVec to contractable_list ###")
        contractable_list.insert(0, EdgeVec(end_vecs[0], is_left_vec=True))
        contractable_list.append(EdgeVec(end_vecs[1], is_left_vec=False))
        #print("### append EdgeVec to contractable_list  doen ###")

        #print("### apply contractable_list into ContractableList ###")
        contractable_list = ContractableList(contractable_list)
        #print("### try to reduce contractable_list ###")
        output = contractable_list.reduce()
        #print("### try to reduce contractable_list done,here is my output ###")

        return output.tensor
        

    def core_len(self):
        #print("LinearRegion core_len")

        return sum([module.core_len() for module in self.module_list])

    def __len__(self):
        #print("LinearRegion __len__")

        return sum([len(module) for module in self.module_list])

class InputRegion(nn.Module):

    def __init__(self, tensor, use_bias=True, fixed_bias=True, bias_mat=None,
                 ephemeral=False,use_GPU=False,parallel=False):
        super().__init__()
        #print("InputRegion __init__")

        assert len(tensor.shape) == 4
        assert tensor.size(1) == tensor.size(2)
        bond_dim = tensor.size(1)

        if use_bias:
            assert bias_mat is None or isinstance(bias_mat, torch.Tensor)
            bias_mat = torch.eye(bond_dim).unsqueeze(0) if bias_mat is None \
                       else bias_mat

            bias_modes = len(list(bias_mat.shape))
            assert bias_modes in [2, 3]
            if bias_modes == 2:
                bias_mat = bias_mat.unsqueeze(0)

        if ephemeral:
            self.register_buffer(name='tensor', tensor=tensor.contiguous())
            self.register_buffer(name='bias_mat', tensor=bias_mat)
        else:
            self.register_parameter(name='tensor', 
                                    param=nn.Parameter(tensor.contiguous()))
            if fixed_bias:
                self.register_buffer(name='bias_mat', tensor=bias_mat)
            else:
                self.register_parameter(name='bias_mat', 
                                        param=nn.Parameter(bias_mat))

        self.use_bias = use_bias
        self.fixed_bias = fixed_bias
        self.use_GPU = use_GPU
        self.parallel = parallel
        
    def forward(self, input_data):
        #print("InputRegion forward")

        tensor = self.tensor
        #if torch.cuda.is_available():
         #   tensor = tensor.cuda()
          #  input_data = input_data.cuda()
        assert len(input_data.shape) == 3
        assert input_data.size(1) == len(self)
        assert input_data.size(2) == tensor.size(3)

        #if torch.cuda.is_available():
        mats = torch.einsum('slri,bsi->bslr', [tensor, input_data])
        #fmats = torch.einsum('slri,bsi->bslr', [tensor, input_data])

        if self.use_bias:
            bond_dim = tensor.size(1)
            bias_mat = self.bias_mat.unsqueeze(0)
            mats = mats + bias_mat.expand_as(mats)

        # try to speed up 2020/4/27
        if self.parallel:
            while mats.size(1) > 1 :
            #print("mat.shape_first : ",mats.shape)
                num_batch = mats.size(0)
                num_site = mats.size(1)
                bond_dim = mats.size(2)

                if num_site%2 == 0:
                    mats = mats.view(num_batch,num_site//2,2,bond_dim,bond_dim)
                    mat1,mat2 = torch.chunk((mats),2,dim=2)
                    mat1,mat2 = mat1.squeeze(2),mat2.squeeze(2)
                    mats = torch.matmul(mat1,mat2)
                    #print("mats.shape : ",mats.shape)
                else:
                    mats,vec = mats.split((num_site-1,1),dim=1)
                    mats = mats.view(num_batch,num_site//2,2,bond_dim,bond_dim)
                    mat1,mat2 = torch.chunk((mats),2,dim=2)
                    mat1,mat2 = mat1.squeeze(2),mat2.squeeze(2)
                    mats = torch.matmul(mat1,mat2)
                    mats = torch.cat((mats,vec),dim=1)
                    #print("mats.shape : ",mats.shape)
        

        return MatRegion(mats,self.use_GPU)

    def _merge(self, offset):
        #print("InputRegion _merge")

        assert offset in [0, 1]
        num_sites = self.core_len()
        parity = num_sites % 2

        if num_sites == 0:
            return [None]

        if (offset, parity) == (1, 1):
            out_list = [self[0], self[1:]._merge(offset=0)[0]]
        elif (offset, parity) == (1, 0):
            out_list = [self[0], self[1:-1]._merge(offset=0)[0], self[-1]]
        elif (offset, parity) == (0, 1):
            out_list = [self[:-1]._merge(offset=0)[0], self[-1]]

        else:
            tensor = self.tensor
            even_cores, odd_cores = tensor[0::2], tensor[1::2]
            assert len(even_cores) == len(odd_cores)

            # Multiply all pairs of cores, keeping inputs separate
            merged_cores = torch.einsum('slui,surj->slrij', [even_cores,
                                                             odd_cores])
            out_list = [MergedInput(merged_cores)]

        return [x for x in out_list if x is not None]

    def __getitem__(self, key):
        #print("InputRegion __getitem__")
        """
        Returns an InputRegion instance sliced along the site index
        """
        assert isinstance(key, int) or isinstance(key, slice)

        if isinstance(key, slice):
            return InputRegion(self.tensor[key])
        else:
            return InputSite(self.tensor[key])

    def get_norm(self):
        #print("InputRegion get_norm")
        """
        Returns list of the norms of each core in InputRegion
        """
        return [torch.norm(core) for core in self.tensor]

    def rescale_norm(self, scale_list):
        #print("InputRegion rescale_norm")
        """
        Rescales the norm of each core by an amount specified in scale_list

        For the i'th tensor defining a core in InputRegion, we rescale as
        tensor_i <- scale_i * tensor_i, where scale_i = scale_list[i]
        """
        assert len(scale_list) == len(self.tensor)

        for core, scale in zip(self.tensor, scale_list):
            core *= scale

    def core_len(self):
        #print("InputRegion core_len")
        return len(self)

    def __len__(self):
        #print("InputRegion __len__")
        return self.tensor.size(0)

class OutputSite(nn.Module):
    

    def __init__(self, tensor):
        #print("OutputSite __init__")
        super().__init__()
        # Register our tensor as a Pytorch Parameter
        self.register_parameter(name='tensor', 
                                param=nn.Parameter(tensor.contiguous()))

    def forward(self, input_data):
        #print("OutputSite forward")

        return OutputCore(self.tensor)

    def get_norm(self):
        #print("OutputSite get_norm")

        return [torch.norm(self.tensor)]


    def rescale_norm(self, scale):
        #print("OutputSite rescale_norm")
        """
        Rescales the norm of our core by a factor of input `scale`
        """
        if isinstance(scale, list):
            assert len(scale) == 1
            scale = scale[0]

        self.tensor *= scale

    def core_len(self):
        #print("OutputSite core_len")
        return 1

    def __len__(self):
        #print("OutputSite __len__")
        return 0
