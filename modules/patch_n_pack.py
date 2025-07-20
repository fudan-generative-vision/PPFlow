import torch


def is_power_of_four(n):
    if n <= 0:
        return False
    while n % 4 == 0:
        n = n // 4
    return n == 1

def patch_n_pack(x, token_length_list):

    target_token_length = token_length_list[-1]

    x_patch = []
    attention_mask = []

    for i in range(len(token_length_list)):
        x_i = x[i]
        B, N, C = x_i.shape

        # ================== pack short length token =====================
        cur_token_length = x_i.shape[1]
        cur_pack_times = target_token_length // cur_token_length
        x_i = x_i.reshape(B // cur_pack_times, cur_pack_times, N, C)
        x_i = x_i.transpose(1,2)
        x_i = x_i.reshape(B // cur_pack_times, cur_pack_times * N, C)
        x_patch.append(x_i)
        # ================================================================

        # ================== generate attention mask =====================
        mask = torch.ones([x_i.shape[0], x_i.shape[1], x_i.shape[1]], device=x_i.device, dtype=x_i.dtype)
        for j in range(cur_pack_times):
            mask[:, j * N: (j + 1) * N, :j * N] = 0.0  # mask the before blocks of this block, e.g. xoxxx, this mask x
            mask[:, j * N: (j + 1) * N, (j + 1) * N: ] = 0.0 # mask the after blocks of this block, e.g. xoxxx, this mask xxx
        attention_mask.append(mask)
        # ================================================================

    x = torch.cat(x_patch, dim=0)
    del x_patch
    attention_mask = torch.cat(attention_mask, dim=0)

    return x, attention_mask

def un_patch_n_pack(x, x_ratio, x_duration, token_length_list, bz):

    x_list = []
    target_token_length = token_length_list[-1]
    bz_start = 0
    bz_end = 0

    for i in range(len(token_length_list)):
        cur_pack_times = target_token_length // token_length_list[i]
        cur_bz = round((bz * x_ratio[i+1]).item()) // cur_pack_times
        bz_end += cur_bz

        x_i = x[bz_start:bz_end]
        bz_start += cur_bz
        # x invertible opration
        x_i = x_i.reshape(cur_bz, token_length_list[i], cur_pack_times, -1) 
        x_i = x_i.transpose(1, 2)                                    
        x_i = x_i.reshape(cur_bz*cur_pack_times, token_length_list[i], -1)                          
        x_list.append(x_i)

    return x_list
    

def pack_pos_emb(pos_emb, bz, x_duration, token_length_list):
    
    target_token_length = token_length_list[-1]

    x_patch = []
    for i in range(len(token_length_list)):
        x_i = pos_emb[i]
        x_i = x_i.repeat(round(((x_duration[i+1] - x_duration[i])*bz).item()), 1, 1)
        B, N, C = x_i.shape

        # ========= interpolation and pack short length pos_emb =========
        cur_token_length = token_length_list[i]
        cur_pack_times = target_token_length // cur_token_length
        x_i = x_i.reshape(B // cur_pack_times, cur_pack_times, N, C)
        x_i = x_i.transpose(1, 2)
        x_i = x_i.reshape(B // cur_pack_times, N * cur_pack_times, C)
        x_patch.append(x_i)
        # ================================================================
    pos_emb = torch.cat(x_patch, dim=0)

    return pos_emb

def expand_tensor(c, token_length, bz_start, bz_end, cur_pack_times):
    B = c.shape[0]
    new_B = bz_end - bz_start

    c = c[bz_start:bz_end].unsqueeze(1).expand(-1, token_length, -1)
    c = c.reshape(new_B // cur_pack_times, cur_pack_times, token_length, -1)
    c = c.transpose(1,2)
    c = c.reshape(new_B // cur_pack_times, cur_pack_times * token_length, -1)

    return c


def pack_shift_scale_gate(c, x_duration, token_length_list, final=False):

    target_token_length = token_length_list[-1]

    c_patch = []

    for i in range(len(token_length_list)):
        B = c.shape[0]
        bz_start, bz_end = round((B*x_duration[i]).item()), round((B*x_duration[i+1]).item())

        cur_pack_times = target_token_length // token_length_list[i]

        # ================== pack short length token =====================
        cur_token_length = token_length_list[i]
        c_i = expand_tensor(c, cur_token_length, bz_start, bz_end, cur_pack_times)
        
        c_patch.append(c_i)
        # ================================================================

    c = torch.cat(c_patch, dim=0)
    del c_patch
    return c.chunk(6, dim=2) if not final else c.chunk(2, dim=2)