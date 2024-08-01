import torch

class Contractable:
#2

    global_bs = None

    def __init__(self, tensor, bond_str):
        #print("Contractable __init__")
        shape = list(tensor.shape)
        num_dim = len(shape)
        str_len = len(bond_str)

        global_bs = Contractable.global_bs
        batch_dim = tensor.size(0)

        if ('b' not in bond_str and str_len == num_dim) or \
           ('b' == bond_str[0] and str_len == num_dim + 1):
            if global_bs is not None:
                tensor = tensor.unsqueeze(0).expand([global_bs] + shape)
            else:
                raise RuntimeError("No batch size given and no previous "
                                   "batch size set")
            if bond_str[0] != 'b':
                bond_str = 'b' + bond_str


        elif global_bs is None or global_bs != batch_dim:
            Contractable.global_bs = batch_dim
        

        self.tensor = tensor
        self.bond_str = bond_str
        #print('self.tensor.device 1 : ',self.tensor.device)
        

    def __mul__(self, contractable, rmul=False):
        #print("Contractable __mul__")

        if isinstance(contractable, Scalar) or \
           not hasattr(contractable, 'tensor') or \
           type(contractable) is MatRegion:
            return NotImplemented

        tensors = [self.tensor, contractable.tensor]
        bond_strs = [list(self.bond_str), list(contractable.bond_str)]
        lowercases = [chr(c) for c in range(ord('a'), ord('z')+1)]


        if rmul:
            tensors = tensors[::-1]
            bond_strs = bond_strs[::-1]


        for i, bs in enumerate(bond_strs):
            assert bs[0] == 'b'
            assert len(set(bs)) == len(bs)
            assert all([c in lowercases for c in bs])
            assert (i == 0 and 'r' in bs) or (i == 1 and 'l' in bs)


        used_chars = set(bond_strs[0]).union(bond_strs[1])
        free_chars = [c for c in lowercases if c not in used_chars]

        specials = ['b', 'l', 'r']
        for i, c in enumerate(bond_strs[1]):
            if c in bond_strs[0] and c not in specials:
                bond_strs[1][i] = free_chars.pop()

        sum_char = free_chars.pop()
        bond_strs[0][bond_strs[0].index('r')] = sum_char
        bond_strs[1][bond_strs[1].index('l')] = sum_char
        specials.append(sum_char)

        out_str = ['b']
        for bs in bond_strs:
            out_str.extend([c for c in bs if c not in specials])
        out_str.append('l' if 'l' in bond_strs[0] else '')
        out_str.append('r' if 'r' in bond_strs[1] else '')

        bond_strs = [''.join(bs) for bs in bond_strs]
        out_str = ''.join(out_str)
        #ein_str = f"{bond_strs[0]},{bond_strs[1]}->{out_str}"
        ein_str = "{},{}->{}".format(bond_strs[0],bond_strs[1],out_str)

        out_tensor = torch.einsum(ein_str, [tensors[0], tensors[1]])

        if out_str == 'br':
            return EdgeVec(out_tensor, is_left_vec=True)
        elif out_str == 'bl':
            return EdgeVec(out_tensor, is_left_vec=False)
        elif out_str == 'blr':
            return SingleMat(out_tensor)
        elif out_str == 'bolr':
            return OutputCore(out_tensor)
        else:
            return Contractable(out_tensor, out_str)

    def __rmul__(self, contractable):
        #print("Contractable __rmul__")

        return self.__mul__(contractable, rmul=True)

    def reduce(self):
        #print("Contractable reduce")

        return self

class ContractableList(Contractable):

    def __init__(self, contractable_list):
        #print("ContractableList __init__")

        self.contractable_list = contractable_list
        #print('self.contractable_list.device : ',self.contractable_list.device)

    def __mul__(self, contractable, rmul=False):
        #print("ContractableList __mul__")
        assert hasattr(contractable, 'tensor')
        output = contractable.tensor
        #print('output.device : ',output.device)

        if rmul:
            for item in self.contractable_list:
                output = item * output
        else:
            for item in self.contractable_list[::-1]:
                output = output * item

        return output

    def __rmul__(self, contractable):
        #print("ContractableList __rmul__")

        return self.__mul__(contractable, rmul=True)

    def reduce(self):
        #print("ContractableList reduce")

        c_list = self.contractable_list
        #print("c_list[1].mats.shape : ",c_list[1].tensor.shape)
        #print('c_list.device : ',c_list.device)
        #print("### start to reduce ###")
        while len(c_list) > 1:
            #print("len(c_list) : {}".format(len(c_list)))
            try:
                #print("c_list[-1] : ",c_list[-1])
                #print("c_list[-2] : ",c_list[-2])
                c_list[-2] = c_list[-2] * c_list[-1]
                del c_list[-1]
            except TypeError:
                #print("c_list[1] : ",c_list[1])
                #print("c_list[0] : ",c_list[0])
                c_list[1] = c_list[0] * c_list[1]
                del c_list[0]
        #print('c_list.device : ',c_list.device)

        return c_list[0]

class MatRegion(Contractable):

    def __init__(self, mats,use_GPU=False):
        #print("MatRegion __init__")
        shape = list(mats.shape)

        super().__init__(mats, bond_str='bslr')
        self.use_GPU =use_GPU

    def __mul__(self, edge_vec, rmul=False):
        #print("MatRegion __mul__")

        if not isinstance(edge_vec, EdgeVec):
            return NotImplemented

        mats = self.tensor
        #print(mats.shape)
        #print('mats.device : ',mats.device)
        num_mats = mats.size(1)
        batch_size = mats.size(0)

        dummy_ind = 1 if rmul else 2
        vec = edge_vec.tensor.unsqueeze(dummy_ind)
        if self.use_GPU:
            vec = vec.cuda()
        #print('vec.device 1: ',vec.device)
        mat_list = [mat.squeeze(1) for mat in torch.chunk(mats, num_mats, 1)]
        
        log_norm = 0
        for i, mat in enumerate(mat_list[::(1 if rmul else -1)], 1):
            if rmul:
                #print("### use rmul {} ###".format(i))
                #print('vec.shape : ',vec.shape)
                #print("mat.shape : ",mat.shape)
                vec = torch.bmm(vec, mat)
            else:
                #print("### use mul {} ###".format(i))
                #print('vec.shape : ',vec.shape)
                #print("mat.shape : ",mat.shape)
                vec = torch.bmm(mat, vec)
        #print('vec.device : ',vec.device)

        return EdgeVec(vec.squeeze(dummy_ind), is_left_vec=rmul)

    def __rmul__(self, edge_vec):
        #print("MatRegion __rmul__")
        return self.__mul__(edge_vec, rmul=True)

    '''def reduce(self):

        mats = self.tensor
        #print('mats.device : ',mats.device)
        shape = list(mats.shape)
        batch_size = mats.size(0)
        size, D = shape[1:3]

        while size > 1:
            odd_size = (size % 2 == 1)
            half_size = size // 2
            nice_size = 2 * half_size
        
            even_mats = mats[:, 0:nice_size:2]
            odd_mats = mats[:, 1:nice_size:2]
            # For odd sizes, set aside one batch of matrices for the next round
            leftover = mats[:, nice_size:]

            # Multiply together all pairs of matrices (except leftovers)
            mats = torch.einsum('bslu,bsur->bslr', [even_mats, odd_mats])
            mats = torch.cat([mats, leftover], 1)

            size = half_size + int(odd_size)

        return SingleMat(mats.squeeze(1))'''

class OutputCore(Contractable):

    def __init__(self, tensor):
        super().__init__(tensor, bond_str='bolr')


class OutputMat(Contractable):

    def __init__(self, mat, is_left_mat):
 
        bond_str = 'b' + ('r' if is_left_mat else 'l') + 'o'
        super().__init__(mat, bond_str=bond_str)

    def __mul__(self, edge_vec, rmul=False):

        if not isinstance(edge_vec, EdgeVec):
            raise NotImplemented
        else:
            return super().__mul__(edge_vec, rmul)

    def __rmul__(self, edge_vec):
        return self.__mul__(edge_vec, rmul=True)

class EdgeVec(Contractable):

    def __init__(self, vec, is_left_vec):
        #print("EdgeVec __inti__")

        bond_str = 'b' + ('r' if is_left_vec else 'l')
        super().__init__(vec, bond_str=bond_str)

    def __mul__(self, right_vec):

        #print("EdgeVec __mul__")

        if not isinstance(right_vec, EdgeVec):
            return NotImplemented
        left_vec = self.tensor.unsqueeze(1)
        right_vec = right_vec.tensor.unsqueeze(2)
        batch_size = left_vec.size(0)
        #print('left_vec.device : ',left_vec.device)
        #print('right_vec.device : ',right_vec.device)
        

        scalar = torch.bmm(left_vec, right_vec).view([batch_size])
        #print('scalar.device : ',scalar.device)

        return Scalar(scalar)

class Scalar(Contractable):

    def __init__(self, scalar):
        #print("Scalar __init__")

        shape = list(scalar.shape)
        if shape is []:
            scalar = scalar.view([1])
            shape = [1]
    
        super().__init__(scalar, bond_str='b')

    def __mul__(self, contractable):
        #print("Scalar __mul__")

        scalar = self.tensor
        tensor = contractable.tensor
        bond_str = contractable.bond_str
        #print('scalar.device : ',scalar.device)
        #print('tensor.device : ',tensor.device)
        

        ein_string = "{},b->{}".format(bond_str,bond_str)
        out_tensor = torch.einsum(ein_string, [tensor, scalar])
        #print('out_tensor.device : ',out_tensor.device)


        contract_class = type(contractable)
        if contract_class is not Contractable:
            return contract_class(out_tensor)
        else:
            return Contractable(out_tensor, bond_str)

    def __rmul__(self, contractable):
        #print("Scalar __rmul__")

        return self.__mul__(contractable)
