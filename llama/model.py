# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Embedding, Linear
import torch

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_len: int=10
    adapter_layer: int=30


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.max_feats = args.max_feats

        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)).cuda()
        self.gate1 = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
        self.gate2 = torch.nn.Parameter(torch.ones(1, self.n_local_heads, 1, 1) * -args.bias)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if adapter is not None:
            adapter_len = adapter.shape[1]
            half_len = int(adapter_len/2)
            adapter_k = self.wk(adapter[:,:half_len,:]).view(1, half_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            adapter_v = self.wv(adapter[:,half_len:,:]).view(1, half_len, self.n_local_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            xk = torch.cat([adapter_k, xk], dim=1)
            xv = torch.cat([adapter_v, xv], dim=1)
            extra_mask = torch.zeros(1, 1, seqlen, half_len).to(mask)
            mask = torch.cat([extra_mask, mask], dim=-1)
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if adapter is not None:            
            adapter_scores = F.softmax(scores[..., :half_len].float(), dim=-1).type_as(xq) * self.gate1.tanh().half()
            if video_start is not None:
                vt_scores = scores[..., half_len:].clone()
                vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] = \
                    vt_scores[:, :, video_start + self.max_feats:, video_start:video_start + self.max_feats] + self.gate2.half()
                vt_scores = F.softmax(vt_scores.float(), dim=-1).type_as(xq)
            else:
                vt_scores = F.softmax(scores[..., half_len:], dim=-1)
            scores = torch.cat([adapter_scores, vt_scores], dim=-1)
        else:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        if adapter is not None:
            return self.wo(output), adapter_k+adapter_v
        else:
            return self.wo(output), 1


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], adapter=None, video_start=None):
        h, v_p = self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, adapter, video_start)
        h = x + h
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out, v_p


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, args):
        super().__init__()
        params.max_feats = args.max_feats
        params.bias = args.bias
        self.args = args
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_feats = args.max_feats


        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.adapter_query = Embedding(params.adapter_len * params.adapter_layer, params.dim)
        self.visual_proj = Linear(768, params.dim, bias=False)
        self.temporal_emb = Embedding(self.max_feats, params.dim)
        self.adapter_len = params.adapter_len
        self.adapter_layer = params.adapter_layer

        self.vqa_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.vaq_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.qav_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.inference_criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

        self.v_at_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.linear_qav = nn.Linear(4096, 768)
        self.norm_qav = RMSNorm(768, eps=params.norm_eps)

        self.vprompt_criterion = nn.KLDivLoss(reduction='none')
        self.linear = Linear(params.dim, 768, bias=False)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

        self.video_label = torch.arange(1, self.max_feats)
        self.tau = args.tau

    def forward(self, data, inference=False):
        video = data['video'].cuda()
        vqa_id, vaq_id, qav_id, negq_id = data['text_id']['vqa'].cuda(), data['text_id']['vaq'].cuda(), data['text_id']['qav'].cuda(), data['text_id']['neg_q'].cuda()
        vqa_label, vaq_label, qav_label = data['label']['vqa'].cuda(), data['label']['vaq'].cuda(), data['label']['qav'].cuda()
        vqa_video_start, vaq_video_start, qav_video_index = data['video_start']['vqa'][0], data['video_start']['vaq'][0], data['video_index']['qav'].cuda()
        
        bsz, n_options, seqlen = vqa_id.shape
        vqa_id, vaq_id, negq_id = vqa_id.reshape(-1, seqlen), vaq_id.reshape(-1, seqlen), negq_id.reshape(-1, 10)
        vqa_label, vaq_label = vqa_label.reshape(-1, seqlen), vaq_label.reshape(-1, seqlen)
        vqa_label, vaq_label = vqa_label[:, 1:].flatten(), vaq_label[:, 1:].flatten()
        
        qav_id = qav_id.reshape(-1, seqlen)
        qav_label = qav_label.reshape(-1, seqlen)
        qav_video_mask = qav_label.ge(0)
        qav_label = qav_label[:, 1:].flatten()
        
        
        with torch.no_grad():
            vqa_h = self.tok_embeddings(vqa_id)
            
            if self.args.vaq and not inference:
                vaq_h = self.tok_embeddings(vaq_id)
                neg_q = self.tok_embeddings(negq_id)
                vaq_h = torch.cat([vaq_h, neg_q], dim=1)
            
            if self.args.qav and not inference:
                qav_h = self.tok_embeddings(qav_id)
            
        freqs_cis = self.freqs_cis.to(vqa_h.device)
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=vqa_h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(vqa_h)

        mask_q = torch.full((1, 1, seqlen+10, seqlen+10), float("-inf"), device=vqa_h.device)
        mask_q = torch.triu(mask_q, diagonal=0 + 1).type_as(vqa_h)

        start_pos = 0
        vaq_loss, qav_loss = torch.tensor([0]).cuda(), torch.tensor([0]).cuda()
        
        adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)
        adapter_v = self.adapter_query.weight.reshape(-1, self.adapter_len, self.params.dim).unsqueeze(1)

        _video_feature = self.visual_proj(video)
        if inference:
            _video_feature = _video_feature.unsqueeze(1).repeat(1, n_options, 1, 1).view(-1, _video_feature.shape[-2], _video_feature.shape[-1])
        video_feature = (_video_feature + self.temporal_emb.weight[None, :, :]).half()
        
        vqa_h = vqa_h.clone()
        vqa_h[:, vqa_video_start:vqa_video_start+self.max_feats] = video_feature

        if self.args.vaq and not inference:
            vaq_h = vaq_h.clone()
            vaq_h[:, vaq_video_start:vaq_video_start+self.max_feats] = video_feature

        if self.args.qav and not inference:
            qav_h = qav_h * ~qav_video_mask[..., None]
            qav_h.scatter_add_(1, qav_video_index[..., None].repeat(1, 1, self.params.dim), video_feature)
        
        for i, layer in enumerate(self.layers[-1 * self.adapter_layer:]):
            if i < 9:#14, 15, 16, 17, 18, 19
                if self.args.vaq and not inference:
                    vaq_h, _ = layer(vaq_h, start_pos, freqs_cis[:seqlen+10], mask_q, adapter[i].half(), vaq_video_start) #vaq_h
                if self.args.qav and not inference:
                    qav_h, _ = layer(qav_h, start_pos, freqs_cis[:seqlen], mask, adapter[i].half(), None)
                vqa_h, _ = layer(vqa_h, start_pos, freqs_cis[:seqlen], mask, adapter[i].half(), vqa_video_start)
            elif  9 <= i < 19: #[20, 21, 22, 23, 24, 25, 26, 27]
                if self.args.vaq and not inference:
                    vaq_h, v_q = layer(vaq_h, start_pos, freqs_cis[:seqlen+10], mask_q, adapter_v[i].half(), vaq_video_start)
                if self.args.qav and not inference:
                    qav_h, qa_v = layer(qav_h, start_pos, freqs_cis[:seqlen], mask, adapter_v[i].half(), None)

                vqa_h, v_a = layer(vqa_h, start_pos, freqs_cis[:seqlen], mask, adapter_v[i].half(), vqa_video_start)
            else:

                vqa_h, _ = layer(vqa_h, start_pos, freqs_cis[:seqlen], mask, None, vqa_video_start)

        # modification of VQA is not easy due to llama archi, hence we use same process of generating vqa_loss to represent torch.cat() of prompt since self.output is shared
        if not inference:
            v_at = torch.zeros([vqa_h.size(0), vqa_h.size(1), vqa_h.size(2)], dtype=torch.float16, device=vqa_h.device)
            v_at[:, vaq_video_start:vaq_video_start+self.max_feats] = v_a.reshape(bsz,10,-1)
            v_at =self.norm(v_at)
            v_at_output = self.output(v_at)
            v_at_output = v_at_output[:, :-1,:].reshape(-1, self.vocab_size)
            va_loss = self.v_at_criterion(v_at_output, vqa_label)

        vqa_h = self.norm(vqa_h)
        vqa_output = self.output(vqa_h)
        vqa_output = vqa_output[:, :-1, :].reshape(-1, self.vocab_size)
        vqa_loss = self.vqa_criterion(vqa_output, vqa_label)

        if self.args.vaq and not inference:
            # his denote cat of v_q prompt
            vqa_h[:, vaq_video_start:vaq_video_start+self.max_feats] = v_q.reshape(bsz, 10, -1)
            #neg_q = vaq_h[:, seqlen:seqlen+10]
            vaq_h = vaq_h[:, :seqlen]
            vaq_mean = torch.mean(vaq_h, dim=1)
            v_q_mean = torch.mean(v_q.reshape(bsz, 10, -1), dim=1) #poivet
            neg_mean = torch.mean(neg_q, dim=1)

            logits_pos = (vaq_mean @ v_q_mean.T) / 1
            logits_negs = (neg_mean @ v_q_mean.T) / 1

            pos = v_q_mean @ vaq_mean.T
            negs = v_q_mean @ neg_mean.T

            targets = F.softmax(
            (pos + negs) / 2 * 1, dim=-1)

            texts_loss = self.cross_entropy(logits_pos, targets, reduction='none')
            images_loss = self.cross_entropy(1-logits_negs, 1-targets, reduction='none')

            contrastive_loss = (images_loss + texts_loss) / 2.0
            contrastive_loss = contrastive_loss.mean()

            vaq_h = self.norm(vaq_h)
            vaq_output = self.output(vaq_h)
            vaq_output = vaq_output[:, :-1, :].reshape(-1, self.vocab_size)
            vaq_loss = self.vaq_criterion(vaq_output, vaq_label)
            
        if self.args.qav and not inference:
            vp = self.W(qa_v.reshape(bsz, 10, -1))
            vp1 = torch.mean(vp[:, :10], dim=1)
            vp_video = torch.mean(video, dim=1).to(torch.float16)

            prompt_v = (vp1 @ vp_video.T) / 1
            video_v = (vp_video @ vp1.T) / 1

            targets_v = F.softmax(
                (prompt_v + video_v) / 2 * 1, dim=-1)
            DYN_loss = self.cross_entropy(prompt_v, targets_v, reduction='mean')

            qav_h = self.norm(qav_h)
            qav_output = torch.bmm(qav_h[:, :-1].float(), _video_feature.transpose(1, 2).float()).reshape(-1, self.max_feats)
            qav_loss = self.qav_criterion(qav_output / self.tau, qav_label)
        
        if inference:
            logits = self.inference_criterion(vqa_output, vqa_label)
            logits = logits.reshape(bsz, n_options, -1)
            return logits
        else:
            return  vqa_loss + (0.02 * va_loss), vaq_loss + (0.05 * contrastive_loss), qav_loss + (0.02 * DYN_loss)

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
