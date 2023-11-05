import torch
import torch.nn as nn
from prettytable import PrettyTable
from torch.nn.modules.activation import Tanh
import copy
import logging
logger = logging.getLogger(__name__)
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)

class BaseModel(nn.Module): 
    def __init__(self, ):
        super().__init__()
        
    def model_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table
 
class CleanCode_Model(BaseModel):

    def __init__(self, base_encoder, args, mlp=False):
        super(CleanCode_Model, self).__init__()

        self.K = args.moco_k
        self.m = args.moco_m
        self.T = args.moco_t
        dim= args.moco_dim

        # create the encoders
        # num_classes is the output fc dimension
        self.code_encoder_q = base_encoder
        self.code_encoder_k = copy.deepcopy(base_encoder)
        self.clean_code_encoder_q = base_encoder
        self.mlp = mlp
        self.time_score= args.time_score
        self.do_whitening = args.do_whitening
        self.do_ineer_loss = args.do_ineer_loss 
        self.agg_way = args.agg_way
        self.args = args

        for param_q, param_k in zip(self.code_encoder_q.parameters(), self.code_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the code queue
        torch.manual_seed(3047)
        torch.cuda.manual_seed(3047)
        self.register_buffer("code_queue", torch.randn(dim,self.K ))
        self.code_queue = nn.functional.normalize(self.code_queue, dim=0)
        self.register_buffer("code_queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the dirty code queue
        self.register_buffer("dirty_code_queue", torch.randn(dim, self.K ))
        self.dirty_code_queue = nn.functional.normalize(self.dirty_code_queue, dim=0)
        self.register_buffer("dirty_code_queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the clean code queue
        self.register_buffer("clean_code_queue", torch.randn(dim, self.K ))
        self.clean_code_queue = nn.functional.normalize(self.clean_code_queue, dim=0)
        self.register_buffer("clean_code_queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        % key encoder的Momentum update
        """
        for param_q, param_k in zip(self.code_encoder_q.parameters(), self.code_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, option='code'):

        batch_size = keys.shape[0]
        if option == 'code':
            code_ptr = int(self.code_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            try:
                self.code_queue[:, code_ptr:code_ptr + batch_size] = keys.T
            except:
                print(code_ptr)
                print(batch_size)
                print(keys.shape)
                exit(111)
            code_ptr = (code_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.code_queue_ptr[0] = code_ptr
        
        elif option == 'dirty_code':
            dirty_code_ptr = int(self.dirty_code_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            try:
                self.dirty_code_queue[:, dirty_code_ptr:dirty_code_ptr + batch_size] = keys.T
            except:
                print(dirty_code_ptr)
                print(batch_size)
                print(keys.shape)
                exit(111)
            dirty_code_ptr = (dirty_code_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.dirty_code_queue_ptr[0] = dirty_code_ptr
        
        elif option == 'clean_code':

            clean_code_ptr = int(self.clean_code_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.clean_code_queue[:, clean_code_ptr:clean_code_ptr + batch_size] = keys.T
            clean_code_ptr = (clean_code_ptr + batch_size) % self.K  # move pointer  ptr->pointer

            self.clean_code_queue_ptr[0] = clean_code_ptr

    

    def forward(self, source_code_q, source_code_k, clean_code_q):
        """
        Input:
            source_code_q: a batch of source codes
            source_code_k: a batch of dirty codes
            clean_code_q: a batch of clean codes
        Output:
            logits, targets        """
        
        # compute query features for source code
        outputs = self.code_encoder_q(source_code_q, attention_mask=source_code_q.ne(1))[0]
        code_q  = (outputs*source_code_q.ne(1)[:,:,None]).sum(1)/source_code_q.ne(1).sum(-1)[:,None]
        code_q  = torch.nn.functional.normalize(code_q, p=2, dim=1)
        # compute query features for clean code
        outputs= self.clean_code_encoder_q(clean_code_q, attention_mask=clean_code_q.ne(1))[0]  # queries: NxC   bs*feature_dim
        clean_code_q = (outputs*clean_code_q.ne(1)[:,:,None]).sum(1)/clean_code_q.ne(1).sum(-1)[:,None]
        clean_code_q = torch.nn.functional.normalize(clean_code_q, p=2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # dirty code
            outputs = self.code_encoder_k(source_code_k, attention_mask=source_code_k.ne(1))[0]  # keys: NxC
            code_k  = (outputs*source_code_k.ne(1)[:,:,None]).sum(1)/source_code_k.ne(1).sum(-1)[:,None]
            code_k  = torch.nn.functional.normalize( code_k, p=2, dim=1)


        ## code vs clean code
        code2clcode_pos = torch.einsum('nc,bc->nb', [code_q, clean_code_q])
        # negative logits: NxK
        code2clcode_neg = torch.einsum('nc,ck->nk', [code_q, self.clean_code_queue.clone().detach()])
        # logits: Nx(n+K)
        code2clcode_logits = torch.cat([self.time_score*code2clcode_pos, code2clcode_neg], dim=1)
        # apply temperature
        code2clcode_logits /= self.T
        # label
        code2clcode_label = torch.arange(code2clcode_logits.size(0), device=code2clcode_logits.device)

        ## clean code vs code
        clcode2code_pos = torch.einsum('nc,bc->nb', [clean_code_q, code_q])
        # negative logits: bsxK
        clcode2code_neg = torch.einsum('nc,ck->nk', [clean_code_q, self.code_queue.clone().detach()])
        # clcode2code_logits: bsx(n+K)
        clcode2code_logits = torch.cat([self.time_score*clcode2code_pos, clcode2code_neg], dim=1)
        # apply temperature
        clcode2code_logits /= self.T
        # label
        clcode2code_label = torch.arange(clcode2code_logits.size(0), device=clcode2code_logits.device)

        ## clean code vs dirty code
        clcode2dtcode_pos = torch.einsum('nc,bc->nb', [clean_code_q, code_k])
        # negative logits: bsxK
        clcode2dtcode_neg = torch.einsum('nc,ck->nk', [clean_code_q, self.dirty_code_queue.clone().detach()])
        # clcode2code_logits: bsx(n+K)
        clcode2dtcode_logits = torch.cat([self.time_score*clcode2dtcode_pos, clcode2dtcode_neg], dim=1)
        # apply temperature
        clcode2dtcode_logits /= self.T
        # label
        clcode2dtcode_label = torch.arange(clcode2dtcode_logits.size(0), device=clcode2dtcode_logits.device)
        
        #logit 3*bsx(1+K)
        inter_logits = torch.cat((code2clcode_logits, clcode2code_logits ,clcode2dtcode_logits ), dim=0)
        # labels: positive key indicators
        inter_labels =  torch.cat((code2clcode_label, clcode2code_label, clcode2dtcode_label), dim=0)

        # dequeue and enqueue
        self._dequeue_and_enqueue(code_q, option='code')
        self._dequeue_and_enqueue(clean_code_q, option='clean_code')
        self._dequeue_and_enqueue(code_k, option='dirty_code')

        return inter_logits, inter_labels, code_q, clean_code_q 

