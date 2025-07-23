from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import _flash_attn_forward
from flash_attn.flash_attn_interface import _flash_attn_backward
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

from einops import rearrange, einsum


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out_dtype = out.dtype
        out, block_out = out.to(torch.float32), block_out.to(torch.float32)
        new_lse = lse - F.logsigmoid(lse - block_lse)
        out = out - F.sigmoid(block_lse - lse).transpose(1, 2)[..., None] * (out - block_out)
    return out.to(out_dtype), new_lse


class FlashAttentionWithMetaToken(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q1,
        q2,
        k1,
        k2,
        v1,
        v2,
        num_meta_tokens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        attn_mask,
        deterministic,
        return_attn_probs,
    ):
        Lq, Lk = q1.shape[1], k1.shape[1]
        if softmax_scale is None:
            softmax_scale = q1.shape[-1] ** (-0.5)
        assert causal and window_size[1] in [0, -1]
        assert num_meta_tokens == k2.shape[1]
        if q2 is not None:
            assert Lq == Lk, "meta_tokens' query doesn't support decoding"
            assert num_meta_tokens == q2.shape[1]

        out1 = _flash_attn_forward(
            q1,
            k1,
            v1,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(window_size[0] - 1, -1),
            alibi_slopes=alibi_slopes,
            attn_mask=attn_mask,
            return_softmax=return_attn_probs
        )
        out1, lse1 = out1[0], out1[5]

        q = torch.cat((q1, q2), dim=1) if q2 is not None else q1
        out2 = _flash_attn_forward(
            q,
            k2,
            v2,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=alibi_slopes,
            attn_mask=attn_mask,
            return_softmax=return_attn_probs
        )
        out2, lse2 = out2[0], out2[5]
        out, lse = _update_out_and_lse(out1, lse1, out2[:,:Lq], lse2[:,:,:Lq])
        ctx.save_for_backward(q1, q2, k1, k2, v1, v2, out, lse, out2[:,Lq:], lse2[:,:,Lq:])
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = True
        ctx.sliding_left = window_size[0] - 1
        ctx.alibi_slopes = alibi_slopes
        ctx.attn_mask = attn_mask
        ctx.deterministic = deterministic

        out = torch.cat((out, out2[:, Lq:]), dim=1) if q2 is not None else out
        return out

    @staticmethod
    def backward(ctx, dout):
        q1, q2, k1, k2, v1, v2, out, lse, out2, lse2 = ctx.saved_tensors
        Lq = q1.shape[1]

        dq1 = torch.zeros_like(q1).contiguous()
        dk1 = torch.zeros_like(k1).contiguous()
        dk2 = torch.zeros_like(k2).contiguous()
        dv1 = torch.zeros_like(v1).contiguous()
        dv2 = torch.zeros_like(v2).contiguous()

        _flash_attn_backward(
            dout[:,:Lq],
            q1,
            k1,
            v1,
            out,
            lse,
            dq1,
            dk1,
            dv1,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=True,
            window_size=(ctx.sliding_left, 0),
            alibi_slopes=ctx.alibi_slopes,
            attn_mask=ctx.attn_mask,
            deterministic=ctx.deterministic
        )
        
        if q2 is not None:
            q = torch.cat((q1, q2), dim=1)
            out = torch.cat((out, out2), dim=1)
            lse = torch.cat((lse, lse2), dim=-1)
        else:
            q = q1
        dq2 = torch.zeros_like(q).contiguous()

        _flash_attn_backward(
            dout,
            q,
            k2,
            v2,
            out,
            lse,
            dq2,
            dk2,
            dv2,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=ctx.alibi_slopes,
            attn_mask=ctx.attn_mask,
            deterministic=ctx.deterministic
        )

        with torch.no_grad():
            dq1 = dq1 + dq2[:,:Lq]
            dq2 = dq2[:,Lq:] if q2 is not None else None

        return dq1, dq2, dk1, dk2, dv1, dv2, None, None, None


class FlashAttentionVarlenWithMetaToken(torch.autograd.Function):

    @staticmethod
    def forward( 
        ctx,
        q1,
        q2,
        k1,
        k2,
        v1,
        v2,
        cu_seqlen_q,
        cu_seqlen_k,
        max_seqlen_q,
        max_seqlen_k,
        num_meta_tokens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        attn_mask,
        deterministic,
        return_attn_probs,
    ):
        Lq, Lk = q1.shape[1], k1.shape[1]
        if softmax_scale is None:
            softmax_scale = q1.shape[-1] ** (-0.5)
        assert causal and window_size[1] in [0, -1]
        assert num_meta_tokens == k2.shape[1]
        if q2 is not None:
            assert Lq == Lk, "meta_tokens' query doesn't support decoding"
            assert num_meta_tokens == q2.shape[1]

        out1 = _flash_attn_varlen_forward(
            q1.squeeze(0),
            k1.squeeze(0),
            v1.squeeze(0),
            cu_seqlens_q=cu_seqlen_q,
            cu_seqlens_k=cu_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(window_size[0] - 1, -1),
            alibi_slopes=alibi_slopes,
            return_softmax=return_attn_probs
        )
        out1, lse1 = out1[0].unsqueeze(0), out1[5]
        lse1 = rearrange(lse1, "b h l -> 1 h (b l)").contiguous()

        q = torch.cat((q1, q2), dim=1) if q2 is not None else q1
        out2 = _flash_attn_forward(
            q,
            k2,
            v2,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False, # ? 官方实现不一样 https://github.com/NVlabs/hymba/blob/main/barebones_hymba/barebones_hymba_block.py
            window_size=(-1, -1),
            alibi_slopes=alibi_slopes,
            attn_mask=attn_mask,
            return_softmax=return_attn_probs
        )
        out2, lse2 = out2[0], out2[5]
        out, lse = _update_out_and_lse(out1, lse1, out2[:,:Lq], lse2[:,:,:Lq])
    
        ctx.save_for_backward(q1, q2, k1, k2, v1, v2, out, lse, out2[:,Lq:], lse2[:,:,Lq:], cu_seqlen_q, cu_seqlen_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.sliding_left = window_size[0] - 1
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = True
        ctx.alibi_slopes = alibi_slopes
        ctx.attn_mask = attn_mask
        ctx.deterministic = deterministic

        out = torch.cat((out, out2[:, Lq:]), dim=1) if q2 is not None else out
        return out

    @staticmethod
    def backward(ctx, dout):
        q1, q2, k1, k2, v1, v2, out, lse, out2, lse2, cu_seqlen_q, cu_seqlen_k = ctx.saved_tensors

        Lq = q1.shape[1]
        dq1 = torch.zeros_like(q1).contiguous().squeeze(0)
        dk1 = torch.zeros_like(k1).contiguous().squeeze(0)
        dv1 = torch.zeros_like(v1).contiguous().squeeze(0)
        _flash_attn_varlen_backward(
            dout[:,:Lq].squeeze(0),
            q1.squeeze(0),
            k1.squeeze(0),
            v1.squeeze(0),
            out.squeeze(0),
            rearrange(lse, "1 h (b l) -> b h l", b = len(cu_seqlen_q)-1).contiguous(),
            dq1,
            dk1,
            dv1,
            cu_seqlen_q,
            cu_seqlen_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=True,
            window_size=(ctx.sliding_left, 0),
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic
        )

        dq1 = dq1.unsqueeze(0)
        dk1 = dk1.unsqueeze(0)
        dv1 = dv1.unsqueeze(0)
        if q2 is not None:
            q = torch.cat((q1, q2), dim=1)
            out = torch.cat((out, out2), dim=1)
            lse = torch.cat((lse, lse2), dim=-1)
        else:
            q = q1
        dq2 = torch.zeros_like(q).contiguous()
        dk2 = torch.zeros_like(k2).contiguous()
        dv2 = torch.zeros_like(v2).contiguous()

        _flash_attn_backward(
            dout,
            q,
            k2,
            v2,
            out,
            lse,
            dq2,
            dk2,
            dv2,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=ctx.alibi_slopes,
            attn_mask=ctx.attn_mask,
            deterministic=ctx.deterministic
        )

        with torch.no_grad():
            dq1 = dq1 + dq2[:,:Lq]
            dq2 = dq2[:,Lq:] if q2 is not None else None
    
        return dq1, dq2, dk1, dk2, dv1, dv2, None, None, None, None, None, None, None


def metatoken_flash_attn_func(
    q1,
    q2,
    k1,
    k2,
    v1,
    v2,
    num_meta_tokens=128,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    attn_mask=None,
):
    return FlashAttentionWithMetaToken.apply(
        q1,
        q2,
        k1,
        k2,
        v1,
        v2,
        num_meta_tokens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        attn_mask,
        deterministic,
        return_attn_probs,
    )


def metatoken_flash_attn_varlen_func(
    q1,
    q2,
    k1,
    k2,
    v1,
    v2,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    num_meta_tokens=128,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    attn_mask=None,
):
    return FlashAttentionVarlenWithMetaToken.apply(
        q1,
        q2,
        k1,
        k2,
        v1,
        v2,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_meta_tokens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        attn_mask,
        deterministic,
        return_attn_probs,
    )


def naive_metatoken_flash_attn(
    q1,
    q2,
    k1,
    k2,
    v1,
    v2,
    softmax_scale=None,
    num_meta_tokens=128,
    sliding_window=(-1, 0),
):
    Lq, Lk = q1.shape[1], k1.shape[1]
    q_id = Lk - Lq + torch.arange(0, Lq)
    k_id = torch.arange(0, Lk)
    causal_mask = q_id[:, None] >= k_id[None, :]
    sliding_mask = (q_id[:, None] - k_id[None, :]) < sliding_window[0]
    mask = torch.ones((Lq, Lk + num_meta_tokens), dtype=torch.bool)
    mask[:, num_meta_tokens:] = causal_mask & sliding_mask

    if q2 is not None:
        q = torch.cat((q1, q2), dim=1)
        mask_meta = torch.cat((torch.ones(num_meta_tokens, num_meta_tokens, dtype=torch.bool), \
                               torch.zeros(num_meta_tokens, Lk, dtype=torch.bool)), dim=1)
        mask = torch.cat((mask, mask_meta), dim=0)
    else:
        q = q1

    k = torch.cat([k2, k1], 1)
    v = torch.cat([v2, v1], 1)
    qk = einsum(q, k, "b q h d, b k h d -> b h q k") * softmax_scale
    qk = torch.where(rearrange(mask.cuda(), "q k -> 1 1 q k"), qk, -float("inf"))
    weight = torch.softmax(qk, dim=-1)
    out = einsum(weight, v, "b h q k, b k h d -> b q h d")
    return out
