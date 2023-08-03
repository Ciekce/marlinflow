import torch

from dataloader import Batch, InputFeatureSet

class PerspectiveNet(torch.nn.Module):
    """
    A "Perspective Network".
    Uses a linear layer to transform the board into a vector of size ft_out.
    It does this for two copies of the board, one from the perspective of the side to move,
    and one from the perspective of the side not to move.
    It then concatenates the two vectors, side to move first, activates them with clipped ReLU,
    and then passes them through a final linear layer to get the evaluation.
    """

    def __init__(self, ft_out: int):
        super().__init__()
        self.perspective = torch.nn.Linear(768, ft_out)
        self.out = torch.nn.Linear(ft_out * 2, 1)

    def forward(self, batch: Batch):
        stm_indices = batch.stm_indices.reshape(-1, 2).T
        nstm_indices = batch.nstm_indices.reshape(-1, 2).T
        board_stm_sparse = torch.sparse_coo_tensor(
            stm_indices, batch.values, (batch.size, 768)
        ).to_dense()
        board_nstm_sparse = torch.sparse_coo_tensor(
            nstm_indices, batch.values, (batch.size, 768)
        ).to_dense()

        stm_pov = self.perspective(board_stm_sparse)
        nstm_pov = self.perspective(board_nstm_sparse)

        hidden = torch.clamp(torch.cat((stm_pov, nstm_pov), dim=1), 0, 1)

        return torch.sigmoid(self.out(hidden))

    def input_feature_set(self) -> InputFeatureSet:
        return InputFeatureSet.BOARD_768

class SquaredPerspectiveNet(torch.nn.Module):
    """
    The same as PerspectiveNet, but the activations of clipped ReLU are squared.
    """

    def __init__(self, ft_out: int):
        super().__init__()
        self.perspective = torch.nn.Linear(768, ft_out)
        self.out = torch.nn.Linear(ft_out * 2, 1)

    def forward(self, batch: Batch):
        stm_indices = batch.stm_indices.reshape(-1, 2).T
        nstm_indices = batch.nstm_indices.reshape(-1, 2).T
        board_stm_sparse = torch.sparse_coo_tensor(
            stm_indices, batch.values, (batch.size, 768)
        ).to_dense()
        board_nstm_sparse = torch.sparse_coo_tensor(
            nstm_indices, batch.values, (batch.size, 768)
        ).to_dense()

        stm_pov = self.perspective(board_stm_sparse)
        nstm_pov = self.perspective(board_nstm_sparse)

        x = torch.clamp(torch.cat((stm_pov, nstm_pov), dim=1), 0, 1)
        hidden = x * x

        return torch.sigmoid(self.out(hidden))

    def input_feature_set(self) -> InputFeatureSet:
        return InputFeatureSet.BOARD_768

class DeepPerspectiveNet(torch.nn.Module):
    def __init__(self, ft_out: int, layer_2: int):
        super().__init__()
        self.perspective = torch.nn.Linear(768, ft_out)
        self.l2 = torch.nn.Linear(ft_out * 2, layer_2)
        self.out = torch.nn.Linear(layer_2, 1)

    def forward(self, batch: Batch):
        stm_indices = batch.stm_indices.reshape(-1, 2).T
        nstm_indices = batch.nstm_indices.reshape(-1, 2).T
        board_stm_sparse = torch.sparse_coo_tensor(
            stm_indices, batch.values, (batch.size, 768)
        ).to_dense()
        board_nstm_sparse = torch.sparse_coo_tensor(
            nstm_indices, batch.values, (batch.size, 768)
        ).to_dense()

        stm_pov = self.perspective(board_stm_sparse)
        nstm_pov = self.perspective(board_nstm_sparse)

        x = torch.clamp(torch.cat((stm_pov, nstm_pov), dim=1), 0, 1)
        x = x * x
        x = torch.clamp(self.l2(x), 0, 1)

        return torch.sigmoid(self.out(x))

    def input_feature_set(self) -> InputFeatureSet:
        return InputFeatureSet.BOARD_768


class HalfKPNet(torch.nn.Module):
    """
    Uses king buckets to choose subnets.
    Features are of the form (our_king_sq, piece_sq, piece_type, piece_colour),
    and does not include the enemy king. (so piece_type is never a king)
    """
    def __init__(self, ft_out: int):
        super().__init__()
        self.ft = torch.nn.Linear(40960, ft_out)
        self.fft = torch.nn.Linear(640, ft_out)
        self.out = torch.nn.Linear(ft_out * 2, 1)

    def forward(self, batch: Batch):

        stm_indices = batch.stm_indices.reshape(-1, 2).T
        nstm_indices = batch.nstm_indices.reshape(-1, 2).T
        board_stm_sparse = torch.sparse_coo_tensor(
            stm_indices, batch.values, (batch.size, 40960)
        )
        board_nstm_sparse = torch.sparse_coo_tensor(
            nstm_indices, batch.values, (batch.size, 40960)
        )

        v_stm_indices = torch.clone(stm_indices)
        v_nstm_indices = torch.clone(nstm_indices)
        v_stm_indices[1][:] %= 640
        v_nstm_indices[1][:] %= 640
        v_board_stm_sparse = torch.sparse_coo_tensor(
            v_stm_indices, batch.values, (batch.size, 640)
        ).to_dense()
        v_board_nstm_sparse = torch.sparse_coo_tensor(
            v_nstm_indices, batch.values, (batch.size, 640)
        ).to_dense()

        stm_ft = self.ft(board_stm_sparse) + self.fft(v_board_stm_sparse)
        nstm_ft = self.ft(board_nstm_sparse) + self.fft(v_board_nstm_sparse)

        hidden = torch.clamp(torch.cat((stm_ft, nstm_ft), dim=1), 0, 1)

        return torch.sigmoid(self.out(hidden))

    def input_feature_set(self) -> InputFeatureSet:
        return InputFeatureSet.HALF_KP


class HalfKANet(torch.nn.Module):
    """
    Uses king buckets to choose subnets.
    Features are of the form (our_king_sq, piece_sq, piece_type, piece_colour),
    does include the enemy king. (so piece_type can be a king)
    """
    def __init__(self, ft_out: int):
        super().__init__()
        # the bucketed feature transformer (768 * 64 = 49152)
        self.perspective = torch.nn.Linear(49152, ft_out)
        # the factoriser - helps with learning by generalising across buckets
        self.factoriser = torch.nn.Linear(768, ft_out)
        # the final layer
        self.out = torch.nn.Linear(ft_out * 2, 1)

    def forward(self, batch: Batch):
        stm_indices = batch.stm_indices.reshape(-1, 2).T
        nstm_indices = batch.nstm_indices.reshape(-1, 2).T
        board_stm_sparse = torch.sparse_coo_tensor(
            stm_indices, batch.values, (batch.size, 49152)
        )
        board_nstm_sparse = torch.sparse_coo_tensor(
            nstm_indices, batch.values, (batch.size, 49152)
        )

        # create a version of the features that ignores the king position,
        # to feed into the factoriser
        unbucketed_stm_indices = torch.clone(stm_indices)
        unbucketed_nstm_indices = torch.clone(nstm_indices)
        unbucketed_stm_indices[1][:] %= 768
        unbucketed_nstm_indices[1][:] %= 768
        unbucketed_board_stm_sparse = torch.sparse_coo_tensor(
            unbucketed_stm_indices, batch.values, (batch.size, 768)
        ).to_dense()
        unbucketed_board_nstm_sparse = torch.sparse_coo_tensor(
            unbucketed_nstm_indices, batch.values, (batch.size, 768)
        ).to_dense()

        # pass through the bucketed feature transformer and the factoriser
        # these are linear layers, so the factoriser could be removed -
        # it's only here to help with learning.
        stm_ft = self.perspective(board_stm_sparse) + self.factoriser(unbucketed_board_stm_sparse)
        nstm_ft = self.perspective(board_nstm_sparse) + self.factoriser(unbucketed_board_nstm_sparse)

        # concatenate the two vectors, side to move first, and
        # activate with squared clipped ReLU.
        x = torch.clamp(torch.cat((stm_ft, nstm_ft), dim=1), 0, 1)
        hidden = x * x

        return torch.sigmoid(self.out(hidden))

    def input_feature_set(self) -> InputFeatureSet:
        return InputFeatureSet.HALF_KA


class NnBoard768Cuda(torch.nn.Module):
    def __init__(self, ft_out: int):
        from cudasparse import DoubleFeatureTransformerSlice

        super().__init__()
        self.max_features = InputFeatureSet.BOARD_768_CUDA.max_features()
        self.ft = DoubleFeatureTransformerSlice(768, ft_out)
        self.out = torch.nn.Linear(ft_out * 2, 1)

    def forward(self, batch: Batch):
        values = batch.values.reshape(-1, self.max_features)
        stm_indices = batch.stm_indices.reshape(-1, self.max_features).type(
            dtype=torch.int32
        )
        nstm_indices = batch.nstm_indices.reshape(-1, self.max_features).type(
            dtype=torch.int32
        )
        stm_ft, nstm_ft = self.ft(
            stm_indices,
            values,
            nstm_indices,
            values,
        )

        hidden = torch.clamp(torch.cat((stm_ft, nstm_ft), dim=1), 0, 1)

        return torch.sigmoid(self.out(hidden))

    def input_feature_set(self) -> InputFeatureSet:
        return InputFeatureSet.BOARD_768_CUDA


class NnHalfKPCuda(torch.nn.Module):
    def __init__(self, ft_out: int):
        super().__init__()
        from cudasparse import DoubleFeatureTransformerSlice

        self.max_features = InputFeatureSet.HALF_KP_CUDA.max_features()
        self.ft = DoubleFeatureTransformerSlice(40960, ft_out)
        self.fft = DoubleFeatureTransformerSlice(640, ft_out)
        self.out = torch.nn.Linear(ft_out * 2, 1)

    def forward(self, batch: Batch):
        values = batch.values.reshape(-1, self.max_features)
        stm_indices = batch.stm_indices.reshape(-1, self.max_features).type(
            dtype=torch.int32
        )
        nstm_indices = batch.nstm_indices.reshape(-1, self.max_features).type(
            dtype=torch.int32
        )
        stm_ft, nstm_ft = self.ft(
            stm_indices,
            values,
            nstm_indices,
            values,
        )
        v_stm_ft, v_nstm_ft = self.fft(
            stm_indices.fmod(640), values, nstm_indices.fmod(640), values
        )

        hidden = torch.clamp(
            torch.cat((stm_ft + v_stm_ft, nstm_ft + v_nstm_ft), dim=1), 0, 1
        )

        return torch.sigmoid(self.out(hidden))

    def input_feature_set(self) -> InputFeatureSet:
        return InputFeatureSet.HALF_KP_CUDA


class NnHalfKACuda(torch.nn.Module):
    def __init__(self, ft_out: int):
        super().__init__()
        from cudasparse import DoubleFeatureTransformerSlice

        self.max_features = InputFeatureSet.HALF_KA_CUDA.max_features()
        self.ft = DoubleFeatureTransformerSlice(49152, ft_out)
        self.fft = DoubleFeatureTransformerSlice(768, ft_out)
        self.out = torch.nn.Linear(ft_out * 2, 1)

    def forward(self, batch: Batch):
        values = batch.values.reshape(-1, self.max_features)
        stm_indices = batch.stm_indices.reshape(-1, self.max_features).type(
            dtype=torch.int32
        )
        nstm_indices = batch.nstm_indices.reshape(-1, self.max_features).type(
            dtype=torch.int32
        )
        stm_ft, nstm_ft = self.ft(
            stm_indices,
            values,
            nstm_indices,
            values,
        )
        v_stm_ft, v_nstm_ft = self.fft(
            stm_indices.fmod(768), values, nstm_indices.fmod(768), values
        )

        hidden = torch.clamp(
            torch.cat((stm_ft + v_stm_ft, nstm_ft + v_nstm_ft), dim=1), 0, 1
        )

        return torch.sigmoid(self.out(hidden))

    def input_feature_set(self) -> InputFeatureSet:
        return InputFeatureSet.HALF_KA_CUDA
