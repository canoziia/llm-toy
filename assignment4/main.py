import sys
from array import array

# --- 1. 基础设置 ---
sys.setrecursionlimit(20000)

# TT Flag 常量
FLAG_EXACT = 0
FLAG_LOWER = 1
FLAG_UPPER = 2
EMPTY_VAL = -999999999


def solve():
    input_file = "input.txt"
    output_file = "output.txt"

    # --- 2. 输入解析 ---
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw = f.read().split()
    except FileNotFoundError:
        return

    if not raw:
        return
    iterator = iter(raw)
    try:
        next(iterator)  # Skip N
    except StopIteration:
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    rows = []
    for line in lines[1:]:
        p = line.replace(",", " ").split()
        if p:
            rows.append(tuple(int(x) for x in p))

    num_rows = len(rows)
    total_score = sum(sum(r) for r in rows)

    # --- 3. 算法优化 I: 单行博弈 DP 预处理 ---
    # 如果只剩下一行，不需要走复杂的 PVS，直接查 DP 表
    # single_row_dp[row_idx][l][r] = value
    # 为了速度，我们展平成线性数组

    # linear_dp[row_idx][local_idx]
    single_row_dps = []

    for cards in rows:
        L = len(cards)
        # DP[len][l] -> value (r is implicit: l + len - 1)
        # 但为了方便直接映射，我们用和 TT 一样的三角映射逻辑
        # dp_table 存的是: 在该行剩余 cards[l...r] 时，当前先手的最大净胜分

        # 状态总数
        num_states = (L * (L + 1)) // 2
        dp = [0] * num_states

        # 我们采用自底向上的 DP 计算
        # 长度为 1, 2, ..., L

        # 辅助函数：计算 local index
        # Offset logic: offset[l] + (r - l)
        offsets = [0] * (L + 1)
        for i in range(L):
            offsets[i + 1] = offsets[i] + (L - i)

        def get_idx(l, r):
            return offsets[l] + (r - l)

        # Length = 1
        for i in range(L):
            dp[get_idx(i, i)] = cards[i]

        # Length = 2 to L
        for length in range(2, L + 1):
            for l in range(L - length + 1):
                r = l + length - 1

                # 转移方程: max(card[l] - dp[l+1, r], card[r] - dp[l, r-1])
                val_l = cards[l] - dp[get_idx(l + 1, r)]
                val_r = cards[r] - dp[get_idx(l, r - 1)]

                dp[get_idx(l, r)] = val_l if val_l > val_r else val_r

        single_row_dps.append(dp)

    # --- 4. 预计算状态机参数 (三角映射 + 位运算) ---
    row_params = []
    current_mult = 1

    initial_tt_idx = 0
    initial_bit_state = 0

    all_cards_flat = []
    card_offsets = []
    cur_flat_offset = 0

    for i, cards in enumerate(rows):
        L = len(cards)
        all_cards_flat.extend(cards)
        card_offsets.append(cur_flat_offset)
        cur_flat_offset += L

        num_states = (L * (L + 1)) // 2

        # Delta计算逻辑同上
        offsets = [0] * (L + 1)
        for k in range(L):
            offsets[k + 1] = offsets[k] + (L - k)

        delta_l = []
        for k in range(L):
            # Delta Index for L->L+1
            d = (L - k - 1) * current_mult
            delta_l.append(d)

        delta_r = -1 * current_mult

        init_local = L - 1
        initial_tt_idx += init_local * current_mult

        # Bit packing (8 bits per index)
        shift_l = i * 16
        shift_r = shift_l + 8

        initial_bit_state |= 0 << shift_l
        initial_bit_state |= (L - 1) << shift_r

        row_params.append(
            {
                "dl": tuple(delta_l),
                "dr": delta_r,
                "sl": shift_l,
                "sr": shift_r,
                "coff": card_offsets[i],
                "len": L,
                "dp": single_row_dps[i],  # 存入预计算的DP表
                "offsets": tuple(offsets),  # 用于在 search 中快速查 DP
                "mult": current_mult,  # 用于从 global idx 还原 local idx (如果需要)
            }
        )

        current_mult *= num_states

    # TT Table
    tt = array("i", [EMPTY_VAL]) * current_mult
    tt_set = tt.__setitem__

    ALL_CARDS = tuple(all_cards_flat)

    # 展开参数以极速访问
    # (sl, sr, dr, dl_tuple, coff, dp_list, offsets_tuple, mult)
    PARAMS = tuple(
        (
            p["sl"],
            p["sr"],
            p["dr"],
            p["dl"],
            p["coff"],
            p["dp"],
            p["offsets"],
            p["mult"],
        )
        for p in row_params
    )
    ROW_COUNT = num_rows
    MASK_8 = 0xFF

    # 硬编码展开 3行/4行 情况
    P0 = PARAMS[0]
    P1 = PARAMS[1] if ROW_COUNT > 1 else None
    P2 = PARAMS[2] if ROW_COUNT > 2 else None
    P3 = PARAMS[3] if ROW_COUNT > 3 else None

    # --- 5. 核心算法: PVS (NegaScout) ---

    def search(tt_idx, bit_state, alpha, beta):
        # 1. 查表
        entry = tt[tt_idx]
        if entry != EMPTY_VAL:
            flag = entry & 3
            val = entry >> 2
            if flag == FLAG_EXACT:
                return val
            if flag == FLAG_LOWER:
                if val > alpha:
                    alpha = val
            elif flag == FLAG_UPPER:
                if val < beta:
                    beta = val
            if alpha >= beta:
                return val

        # 2. 算法优化: 单行终止检查 (Endgame Database)
        # 检查是否只有一行还有牌 (l <= r)
        # 这是一个 O(Rows) 的检查，但能剪掉深层递归
        active_rows = 0
        last_active_idx = -1
        last_l = -1
        last_r = -1

        # Unrolled check
        l0 = (bit_state >> P0[0]) & MASK_8
        r0 = (bit_state >> P0[1]) & MASK_8
        if l0 <= r0:
            active_rows += 1
            last_active_idx = 0
            last_l, last_r = l0, r0

        if ROW_COUNT > 1:
            l1 = (bit_state >> P1[0]) & MASK_8
            r1 = (bit_state >> P1[1]) & MASK_8
            if l1 <= r1:
                active_rows += 1
                last_active_idx = 1
                last_l, last_r = l1, r1

        if ROW_COUNT > 2:
            l2 = (bit_state >> P2[0]) & MASK_8
            r2 = (bit_state >> P2[1]) & MASK_8
            if l2 <= r2:
                active_rows += 1
                last_active_idx = 2
                last_l, last_r = l2, r2

        if ROW_COUNT > 3:
            l3 = (bit_state >> P3[0]) & MASK_8
            r3 = (bit_state >> P3[1]) & MASK_8
            if l3 <= r3:
                active_rows += 1
                last_active_idx = 3
                last_l, last_r = l3, r3

        # 如果只剩一行，直接返回 DP 结果
        if active_rows == 1:
            p = PARAMS[last_active_idx]
            # local_idx = offsets[l] + (r - l)
            dp_idx = p[6][last_l] + (last_r - last_l)
            val = p[5][dp_idx]
            # 存入 TT (Exact)
            tt_set(tt_idx, (val << 2) | FLAG_EXACT)
            return val

        if active_rows == 0:
            return 0

        # 3. 生成移动 (Move Generation)
        moves = []

        # Row 0
        if l0 <= r0:
            moves.append(
                (ALL_CARDS[P0[4] + l0], tt_idx + P0[3][l0], bit_state + (1 << P0[0]))
            )
            if l0 < r0:
                moves.append(
                    (ALL_CARDS[P0[4] + r0], tt_idx + P0[2], bit_state - (1 << P0[1]))
                )

        # Row 1
        if ROW_COUNT > 1 and l1 <= r1:
            moves.append(
                (ALL_CARDS[P1[4] + l1], tt_idx + P1[3][l1], bit_state + (1 << P1[0]))
            )
            if l1 < r1:
                moves.append(
                    (ALL_CARDS[P1[4] + r1], tt_idx + P1[2], bit_state - (1 << P1[1]))
                )

        # Row 2
        if ROW_COUNT > 2 and l2 <= r2:
            moves.append(
                (ALL_CARDS[P2[4] + l2], tt_idx + P2[3][l2], bit_state + (1 << P2[0]))
            )
            if l2 < r2:
                moves.append(
                    (ALL_CARDS[P2[4] + r2], tt_idx + P2[2], bit_state - (1 << P2[1]))
                )

        # Row 3
        if ROW_COUNT > 3 and l3 <= r3:
            moves.append(
                (ALL_CARDS[P3[4] + l3], tt_idx + P3[3][l3], bit_state + (1 << P3[0]))
            )
            if l3 < r3:
                moves.append(
                    (ALL_CARDS[P3[4] + r3], tt_idx + P3[2], bit_state - (1 << P3[1]))
                )

        # 4. 排序 (Move Ordering)
        # 启发式：拿大牌优先。对于 PVS 至关重要。
        moves.sort(reverse=True)

        # 5. PVS (Principal Variation Search) 递归逻辑
        best_val = -999999999
        original_alpha = alpha

        for i, (card, next_tt, next_bit) in enumerate(moves):
            if i == 0:
                # 对第一个节点（Principal Variation）进行全窗口搜索
                v = card - search(next_tt, next_bit, -beta, -alpha)
            else:
                # 对后续节点进行零窗口搜索 (Null Window Search)
                # 探测是否存在 alpha < v < beta 的情况
                # Window: (-alpha - 1, -alpha) -> 这是一个空窗口
                v = card - search(next_tt, next_bit, -alpha - 1, -alpha)

                # 如果探测失败（Fail High），说明这个节点可能比 PV 节点好，
                # 或者至少比当前 alpha 好，且没有发生截断
                # 必须用全窗口重新搜索以获取精确值
                if alpha < v < beta:
                    v = card - search(next_tt, next_bit, -beta, -alpha)

            if v > best_val:
                best_val = v

            if best_val > alpha:
                alpha = best_val

            if alpha >= beta:
                break

        # 6. 存表
        if best_val <= original_alpha:
            flag = FLAG_UPPER
        elif best_val >= beta:
            flag = FLAG_LOWER
        else:
            flag = FLAG_EXACT

        tt_set(tt_idx, (best_val << 2) | flag)
        return best_val

    # --- 6. 根节点展开 (不使用 PVS，正常初始化) ---
    root_moves = []
    bit_state = initial_bit_state

    for i in range(ROW_COUNT):
        p = PARAMS[i]
        l = (bit_state >> p[0]) & MASK_8
        r = (bit_state >> p[1]) & MASK_8

        if l <= r:
            # Left
            root_moves.append(
                {
                    "val": ALL_CARDS[p[4] + l],
                    "tt": initial_tt_idx + p[3][l],
                    "bit": bit_state + (1 << p[0]),
                    "row": i + 1,
                    "side": "左端",
                }
            )
            # Right
            if l < r:
                root_moves.append(
                    {
                        "val": ALL_CARDS[p[4] + r],
                        "tt": initial_tt_idx + p[2],
                        "bit": bit_state - (1 << p[1]),
                        "row": i + 1,
                        "side": "右端",
                    }
                )

    root_moves.sort(key=lambda x: x["val"], reverse=True)

    best_diff = -float("inf")
    best_move_data = None
    alpha = -float("inf")
    beta = float("inf")

    # 根节点其实也可以应用 PVS 逻辑
    for i, m in enumerate(root_moves):
        if i == 0:
            v = m["val"] - search(m["tt"], m["bit"], -beta, -alpha)
        else:
            v = m["val"] - search(m["tt"], m["bit"], -alpha - 1, -alpha)
            if alpha < v < beta:
                v = m["val"] - search(m["tt"], m["bit"], -beta, -alpha)

        if v > best_diff:
            best_diff = v
            best_move_data = m
        if v > alpha:
            alpha = v

    # --- 7. 输出 ---
    if best_move_data:
        my = (total_score + best_diff) // 2
        opp = total_score - my
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(
                f"第{best_move_data['row']}行 {best_move_data['side']} 牌点数{best_move_data['val']}\n"
            )
            f.write(f"小红: {my} 小蓝: {opp}")


if __name__ == "__main__":
    solve()
