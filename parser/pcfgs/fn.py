from torch.utils.checkpoint import checkpoint as ckp

'''
To save memory of the inside algorithm.
'''


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return ckp(func, *args, **kwargs)

    return wrapper


'''
I borrow the idea from:
https://github.com/yzhangcs/parser/blob/a8e6f443febf8d986cd6eba74966fdf924cb567d/supar/utils/fn.py#L32
to implement `stripe' function. If you do not understand the code, plz refer to their code and comment.
Roughly speaking, this function packs all inside scores of spans of same width w, which facilitates parallel computation.      
'''


def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    else:
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)


def stripe_add_(x, y, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                           stride=stride,
                           storage_offset=(offset[0] * seq_len + offset[1]) * numel)
        x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                     stride=stride,
                     storage_offset=(offset[0] * seq_len + offset[1]) * numel).copy_(tmp + y)
    else:
        tmp = x.as_strided(size=(x.shape[0], n, w),
                           stride=stride,
                           storage_offset=(offset[0] * seq_len + offset[1]) * numel)
        x.as_strided(size=(x.shape[0], n, w),
                     stride=stride,
                     storage_offset=(offset[0] * seq_len + offset[1]) * numel).copy_(tmp + y)


'''
This function is similar to the above one, but additionally select head words, used in lexicalized PCFGs.
'''


def stripe_with_headword(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(x.shape[0], n, w, w + 1, *list(x.shape[4:])),
                        stride=stride,
                        storage_offset=(offset[0] * seq_len + offset[1]) * numel)




def stripe_with_headword_add_(x, y, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel + stride[3]
    stride[2] = (1 if dim == 1 else seq_len) * numel
    tmp = x.as_strided(size=(x.shape[0], n, w, w + 1, *list(x.shape[4:])),
                       stride=stride,
                       storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    x.as_strided(size=(x.shape[0], n, w, w + 1, *list(x.shape[4:])),
                 stride=stride,
                 storage_offset=(offset[0] * seq_len + offset[1]) * numel).copy_(tmp + y)


'''
used in bilexicalized-PCFGs.  
'''


def stripe_grammar_rules(x, n, w):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    return x.as_strided(size=(x.shape[0], n, w, *x.shape[2:]),
                        stride=new_stride, storage_offset=0)


def stripe_grammar_rules_add_(x, y, n, w, offset=0):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1])
    new_stride.append(stride[1])
    new_stride.extend(stride[2:])
    tmp = x.as_strided(size=(x.shape[0], n, w, *x.shape[2:]),
                       stride=new_stride, storage_offset=offset * stride[1])
    x.as_strided(size=(x.shape[0], n, w, *x.shape[2:]),
                 stride=new_stride, storage_offset=offset * stride[1]).copy_(tmp + y)


def diagonal_copy_(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(y)
    else:
        x.as_strided(size=(x.shape[0], seq_len - w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(y)


def diagonal(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )
    else:
        return x.as_strided(size=(x.shape[0], seq_len - w),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )


def diagonal_copy_with_headword(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(y)
    else:
        # new_stride.append(stride[3])
        x.as_strided(size=(x.shape[0], seq_len - w, w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(y)


def diagonal_with_headword(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        return x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     )
    else:
        # new_stride.append(stride[3])
        return x.as_strided(size=(x.shape[0], seq_len - w, w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     )


def diagonal_with_headword_add_(x, y, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    if len(x.shape) > 4:
        new_stride.extend(stride[4:])
        tmp = x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                           stride=new_stride,
                           storage_offset=w * stride[2]
                           )
        x.as_strided(size=(x.shape[0], seq_len - w, w, *list(x.shape[4:])),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(tmp + y)

    else:
        # new_stride.append(stride[3])
        tmp = x.as_strided(size=(x.shape[0], seq_len - w, w),
                           stride=new_stride,
                           storage_offset=w * stride[2]
                           )
        x.as_strided(size=(x.shape[0], seq_len - w, w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(tmp + y)


# the following three functions are used in implementing Eisner-Satta algorithm...
def stripe_headed_left(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, x.shape[3], nt, t),
                        stride=new_stride,
                        storage_offset=w * stride[2] + nt * stride[-1])


def stripe_headed_right(x, n, w, nt, t):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    new_stride.append(stride[1])
    new_stride.extend(stride[3:])
    return x.as_strided(size=(x.shape[0], n, w, x.shape[3], t, nt),
                        stride=new_stride,
                        storage_offset=stride[1] + nt * stride[-2])


def stripe_need_dad(x, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                        storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3])


def stripe_need_dad_add_(x, y, n, w, start, end, headstart):
    x = x.contiguous()
    stride = list(x.stride())
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2] + stride[3])
    new_stride.append(stride[3])
    new_stride.extend(stride[4:])
    tmp = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                       storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3])
    x.as_strided(size=(x.shape[0], n, w, *list(x.shape[4:])), stride=new_stride,
                 storage_offset=start * stride[1] + (end) * stride[2] + headstart * stride[3]).copy_(tmp + y)