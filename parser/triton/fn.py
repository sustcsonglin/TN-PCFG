import pdb
import torch
import triton
import triton.language as tl


@triton.jit
def logaddexp(a, b):
    tmp = a - b
    return tl.where(tmp > 0, tl.log(tl.exp(b - a) + 1) + a, tl.log(tl.exp(a-b) + 1) + b)

@triton.jit
def _kernel_inside_merge(
            alpha_c,            
            out,
            out_noramlized,
            normalizer,           
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            # stride_tmp0, stride_tmp1, stride_tmp2,
            stride_out0, stride_out1,
            stride_normalizer0, stride_normalizer1,
            batch, r,
            # stride_normalizer,
            BLOCK_R1: tl.constexpr,
            w
            ):    
    
    # tmp: [b, n, w, r]
    # out: [b, n, r]
    # normalizer: [b, n, r]
    b_idx = tl.program_id(0) 
    if b_idx >= batch:
        return 

    start = tl.program_id(1)
    end = start + w
    # acc1 = tl.zeros((w-1, BLOCK_R1))

    offset_r = tl.arange(0, BLOCK_R1)

    l_ptr = alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (start+1) * stride_alpha_c3 +  offset_r    
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start +1) * stride_alpha_c2 +  (end) * stride_alpha_c3 + r + offset_r 


    acc = tl.zeros((BLOCK_R1,), dtype=tl.float32) - 1e9

    mask= tl.arange(0, BLOCK_R1) < r

    for _ in range(0, w-1):
        left = tl.load(l_ptr,mask=mask, other=-1e9)
        right = tl.load(r_ptr,mask=mask,other=-1e9)
        merge = left + right
        acc = logaddexp(acc, merge)
        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2

    tl.store(out + b_idx * stride_out0 + start * stride_out1 + offset_r, acc, mask=mask)
    
    acc_max = tl.max(acc, 0)
    tl.store(normalizer + b_idx * stride_normalizer0 + start, acc_max)

    acc2 = tl.exp(acc - acc_max)
    tl.store(out_noramlized + b_idx * stride_out0 + start * stride_out1 + offset_r, acc2, mask=mask)
    


    
@triton.jit
def _kernel_bwd_merge(
            alpha_c,            
            out,
            out_normalized,
            out_grad,
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_out0, stride_out1, 
            batch, r,
            # stride_normalizer,
            BLOCK_R1: tl.constexpr,
            w,
        ):    
    

    b_idx = tl.program_id(0) 

    if b_idx >= batch:
        return 

    start = tl.program_id(1)


    end = start + w

    # acc1 = tl.zeros((w-1, BLOCK_R1))

    offset_r = tl.arange(0, BLOCK_R1)

    l_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + (start+1) * stride_alpha_c2 + (start) * stride_alpha_c3 +  offset_r    
    r_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + (end) * stride_alpha_c2 +  (start+1) * stride_alpha_c3 + r + offset_r 

    l_ptr = alpha_c + b_idx * stride_alpha_c1 + (start) * stride_alpha_c2 + (start+1) * stride_alpha_c3 +  offset_r    
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start+1) * stride_alpha_c2 + (end) * stride_alpha_c3 + r + offset_r    

    mask = tl.arange(0, BLOCK_R1) < r

    do = tl.load(out_normalized + b_idx * stride_out0 + start * stride_out1 +  tl.arange(0, BLOCK_R1), mask=mask, other=0)
    do *= tl.load(out_grad + b_idx * stride_out0 + start * stride_out1 +  tl.arange(0, BLOCK_R1), mask=mask, other=0)

    parent_score = tl.load(out + b_idx * stride_out0 + start * stride_out1 +  tl.arange(0, BLOCK_R1), mask=mask, other=0)

    for _ in range(0, w-1):
        left_score = tl.load(l_ptr, mask=mask, other=0)
        right_score = tl.load(r_ptr, mask=mask, other=0)
        new_grad = tl.exp(left_score + right_score - parent_score) * do
        tl.atomic_add(l_bwd_ptr,  new_grad, mask=mask)
        tl.atomic_add(r_bwd_ptr,  new_grad, mask=mask)        
        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2
        l_bwd_ptr += stride_alpha_c2
        r_bwd_ptr += stride_alpha_c3


@triton.jit
def _kernel_inside_merge_w_split(
            alpha_c,            
            out,
            out_noramlized,
            split,
            normalizer,           
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            # stride_tmp0, stride_tmp1, stride_tmp2,
            stride_out0, stride_out1,
            stride_normalizer0, stride_normalizer1,
            batch, r,
            # stride_normalizer,
            BLOCK_R1: tl.constexpr,
            w
            ):    
    
    # tmp: [b, n, w, r]
    # out: [b, n, r]
    # normalizer: [b, n, r]
    b_idx = tl.program_id(0) 
    if b_idx >= batch:
        return 

    start = tl.program_id(1)
    end = start + w
    # acc1 = tl.zeros((w-1, BLOCK_R1))

    offset_r = tl.arange(0, BLOCK_R1)

    l_ptr = alpha_c + b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (start+1) * stride_alpha_c3 +  offset_r    
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start +1) * stride_alpha_c2 +  (end) * stride_alpha_c3 + r + offset_r 


    acc = tl.zeros((BLOCK_R1,), dtype=tl.float32) - 1e9

    mask= tl.arange(0, BLOCK_R1) < r

    for _ in range(0, w-1):
        left = tl.load(l_ptr,mask=mask, other=-1e9)
        right = tl.load(r_ptr,mask=mask,other=-1e9)
        merge = left + right
        acc = logaddexp(acc, merge)
        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2

    acc = acc + tl.load(split + tl.arange(0, BLOCK_R1), mask=mask, other=0)

    tl.store(out + b_idx * stride_out0 + start * stride_out1 + offset_r, acc, mask=mask)
    
    acc_max = tl.max(acc, 0)
    tl.store(normalizer + b_idx * stride_normalizer0 + start, acc_max)

    acc2 = tl.exp(acc - acc_max)
    tl.store(out_noramlized + b_idx * stride_out0 + start * stride_out1 + offset_r, acc2, mask=mask)
    

    
@triton.jit
def _kernel_bwd_merge_w_split(
            alpha_c,            
            out,
            out_normalized,
            split,
            split_grad,
            out_grad,
            stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
            stride_out0, stride_out1, 
            batch, r,
            # stride_normalizer,
            BLOCK_R1: tl.constexpr,
            w,
        ):    
    

    b_idx = tl.program_id(0) 

    if b_idx >= batch:
        return 

    start = tl.program_id(1)

    end = start + w

    # acc1 = tl.zeros((w-1, BLOCK_R1))

    offset_r = tl.arange(0, BLOCK_R1)

    l_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + (start+1) * stride_alpha_c2 + (start) * stride_alpha_c3 +  offset_r    
    r_bwd_ptr = alpha_c + b_idx * stride_alpha_c1 + (end) * stride_alpha_c2 +  (start+1) * stride_alpha_c3 + r + offset_r 

    l_ptr = alpha_c + b_idx * stride_alpha_c1 + (start) * stride_alpha_c2 + (start+1) * stride_alpha_c3 +  offset_r    
    r_ptr = alpha_c + b_idx * stride_alpha_c1 + (start+1) * stride_alpha_c2 + (end) * stride_alpha_c3 + r + offset_r    

    mask = tl.arange(0, BLOCK_R1) < r

    do = tl.load(out_normalized + b_idx * stride_out0 + start * stride_out1 +  tl.arange(0, BLOCK_R1), mask=mask, other=0)
    do *= tl.load(out_grad + b_idx * stride_out0 + start * stride_out1 +  tl.arange(0, BLOCK_R1), mask=mask, other=0)

    tl.atomic_add(split_grad + tl.arange(0, BLOCK_R1), do, mask=mask)

    parent_score = tl.load(out + b_idx * stride_out0 + start * stride_out1 +  tl.arange(0, BLOCK_R1), mask=mask, other=0) 
    parent_score = parent_score - tl.load(split + tl.arange(0, BLOCK_R1), mask=mask, other=0)

    for _ in range(0, w-1):
        left_score = tl.load(l_ptr, mask=mask, other=0)
        right_score = tl.load(r_ptr, mask=mask, other=0)
        new_grad = tl.exp(left_score + right_score - parent_score) * do
        tl.atomic_add(l_bwd_ptr,  new_grad, mask=mask)
        tl.atomic_add(r_bwd_ptr,  new_grad, mask=mask)        
        l_ptr += stride_alpha_c3
        r_ptr += stride_alpha_c2
        l_bwd_ptr += stride_alpha_c2
        r_bwd_ptr += stride_alpha_c3


@triton.jit
def kernel_log_and_diagonal_copy(
    out,
    normalizer,
    alpha_c,    
    stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
    stride_out0, stride_out1,
    stride_normalizer0, stride_normalizer1,
    batch, r,
    BLOCK_R1: tl.constexpr,
    w
):    

    b_idx = tl.program_id(0) 
    if b_idx >= batch:
        return 
    
    start = tl.program_id(1)    
    mask = tl.arange(0, BLOCK_R1) < r

    x = tl.load(out + b_idx * stride_out0 + start * stride_out1 + tl.arange(0, BLOCK_R1), mask=mask, other=1)

    x_normalizer = tl.load(normalizer + b_idx * stride_normalizer0 + start)

    out_log = tl.log(x + 1e-9)
    out_log = out_log + x_normalizer
    tl.store(alpha_c +  b_idx * stride_alpha_c1 + start * stride_alpha_c2 + (start+w) * stride_alpha_c3 +  tl.arange(0,  BLOCK_R1) , out_log, mask=mask)


@triton.jit
def _bwd_log_and_diagonal_copy(
    out, out_grad,
    alpha_c,    
    stride_alpha_c1, stride_alpha_c2, stride_alpha_c3, 
    stride_out0, stride_out1, 
    batch, r,
    BLOCK_R1: tl.constexpr,
    w
):
    

    b_idx = tl.program_id(0) 
    if b_idx >= batch:
        return 


    mask = tl.arange(0, BLOCK_R1) < r
    start = tl.program_id(1)    
    x = tl.load(out + b_idx * stride_out0 + start * stride_out1 + tl.arange(0, BLOCK_R1), mask=mask, other=1)
    out_log = 1/(x + 1e-9)    

    do = tl.load(alpha_c +  b_idx * stride_alpha_c1 + (start+w) * stride_alpha_c2 + (start) * stride_alpha_c3 +  tl.arange(0,  BLOCK_R1), mask=mask, other=0)

    do *= out_log

    tl.store(out_grad + b_idx * stride_out0 + start * stride_out1 + tl.arange(0, BLOCK_R1), do, mask=mask)




class DIAGONAL_COPY_AND_LOG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, normalizer, alpha_c):
        b, n = out.shape[0], out.shape[1] 
        N = alpha_c.shape[1]
        w = N - n 
        r = int(alpha_c.shape[-1])  * 2

        batch = triton.next_power_of_2(b)

        num_warps = 4
        R = triton.next_power_of_2(r)

        if R >= 2048:
            num_warps = 8
        if R >= 4096:
            num_warps = 16

        kernel_log_and_diagonal_copy[batch, n](out, normalizer,
                                     alpha_c,             
                                     alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
                                     out.stride(0), out.stride(1),
                                     normalizer.stride(0), normalizer.stride(1),
                                     b, r, 
                                     BLOCK_R1=R,
                                     w=w,
                                     num_warps=num_warps
                                     )
        ctx.save_for_backward(out, alpha_c)

        return alpha_c
        
    @staticmethod
    def backward(ctx, do):
        out, alpha_c = ctx.saved_tensors
        b, n = out.shape[0], out.shape[1]  
        N = alpha_c.shape[1]
        w = N - n 
        r = alpha_c.shape[-1]   * 2
        out_grad = out.new_zeros(*out.shape)

        batch = triton.next_power_of_2(b)
        R = triton.next_power_of_2(r)                        


        num_warps = 4


        if R >= 2048:
            num_warps = 8
        if R >= 4096:
            num_warps = 16

        _bwd_log_and_diagonal_copy[batch, n](
            out, out_grad, alpha_c,
            alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
            out.stride(0), out.stride(1), b, r,
            BLOCK_R1=R,
            w=w,
            num_warps=num_warps               
        )
    
        return out_grad,  None, alpha_c



        
        
class MERGE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  normalizer, span_indicator, alpha_c):        
        b, n = normalizer.shape[0], normalizer.shape[1]
        N = alpha_c.shape[1]
        w = N - n
        r = alpha_c.shape[-1]   
        

        out = alpha_c.new_zeros(b, n, r)
        out_normalized =  alpha_c.new_zeros(b, n, r)
        
        batch = triton.next_power_of_2(b)
        
        num_warps = 4
        if r >= 2048:
            num_warps = 8
        if r >= 4096:
            num_warps = 16

        _kernel_inside_merge[batch, n](
            alpha_c,                        
            out,
            out_normalized,
            normalizer,           
            alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
            # tmp.stride(0), tmp.stride(1), tmp.stride(2),
            out.stride(0), out.stride(1),
            normalizer.stride(0), normalizer.stride(1), b, r,          
            # stride_normalizer,            
            BLOCK_R1= triton.next_power_of_2(r),
            w=w,
            num_warps=num_warps
        )

        ctx.save_for_backward(out, out_normalized, alpha_c, span_indicator)                
        return out_normalized, normalizer
            
    @staticmethod
    def backward(ctx, do, do2):

        out, out_normalized, alpha_c, span_indicator = ctx.saved_tensors
        b, n = out.shape[0], out.shape[1]    
        N = alpha_c.shape[1]
        w = N - n 
        r = int(alpha_c.shape[-1])   
        batch = triton.next_power_of_2(b)
    
        num_warps = 4

        if r >= 2048:
            num_warps = 8

        if r >= 4096:
            num_warps = 16

        _kernel_bwd_merge[batch, n](
            alpha_c,                    
            out,
            out_normalized,
            do,
            alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
            out.stride(0), out.stride(1), b,r,
            BLOCK_R1=triton.next_power_of_2(r),
            w=w,
            num_warps=num_warps
        )
        
        grad_indicator = None        
        if span_indicator.requires_grad:
            grad_indicator = alpha_c[:, torch.arange(n) + w, torch.arange(n)].sum([-1, -2])
        
        return None, grad_indicator, alpha_c




class MERGE_w_split_prob(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  normalizer, span_indicator, split, alpha_c):        
        b, n = normalizer.shape[0], normalizer.shape[1]
        N = alpha_c.shape[1]
        w = N - n
        r = alpha_c.shape[-1]   
        
        out = alpha_c.new_zeros(b, n, r)
        out_normalized =  alpha_c.new_zeros(b, n, r)
        
        batch = triton.next_power_of_2(b)
        
        num_warps = 4
        if r >= 2048:
            num_warps = 8
        if r >= 4096:
            num_warps = 16

        _kernel_inside_merge_w_split[batch, n](
            alpha_c,                        
            out,
            out_normalized,
            split,
            normalizer,           
            alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
            # tmp.stride(0), tmp.stride(1), tmp.stride(2),
            out.stride(0), out.stride(1),
            normalizer.stride(0), normalizer.stride(1), b, r,          
            # stride_normalizer,            
            BLOCK_R1= triton.next_power_of_2(r),
            w=w,
            num_warps=num_warps
        )

        ctx.save_for_backward(out, out_normalized, split, alpha_c, span_indicator)                
        return out_normalized, normalizer
            
    @staticmethod
    def backward(ctx, do, do2):

        out, out_normalized, split, alpha_c, span_indicator = ctx.saved_tensors
        split_grad = split.new_zeros(*split.shape)
        b, n = out.shape[0], out.shape[1]    
        N = alpha_c.shape[1]
        w = N - n 
        r = int(alpha_c.shape[-1])   
        batch = triton.next_power_of_2(b)
    
        num_warps = 4

        if r >= 2048:
            num_warps = 8
        if r >= 4096:
            num_warps = 16

        _kernel_bwd_merge_w_split[batch, n](
            alpha_c,                    
            out,
            out_normalized,
            split,
            split_grad,
            do,
            alpha_c.stride(0), alpha_c.stride(1), alpha_c.stride(2), 
            out.stride(0), out.stride(1), b,r,
            BLOCK_R1=triton.next_power_of_2(r),
            w=w,
            num_warps=num_warps
        )
        
        grad_indicator = None
        if span_indicator.requires_grad:
            grad_indicator = alpha_c[:, torch.arange(n) + w, torch.arange(n)].sum([-1, -2])
        
        return None, grad_indicator, split_grad, alpha_c



_log_then_diagonal_copy_ = DIAGONAL_COPY_AND_LOG.apply
_merge = MERGE.apply
_merge_w_split = MERGE_w_split_prob.apply




