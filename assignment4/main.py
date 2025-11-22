import sys
from array import array

# --- 1. 极限性能与环境设置 ---
sys.setrecursionlimit(20000)

# TT Flag 常量
FLAG_EXACT = 0
FLAG_LOWER = 1
FLAG_UPPER = 2
EMPTY_VAL = -999999999  # 哨兵值，表示未填充


def solve():
    input_file = "input.txt"
    output_file = "output.txt"

    # --- 2. 鲁棒的输入读取 ---
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            raw_content = f.read().split()
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    if not raw_content:
        return

    iterator = iter(raw_content)
    try:
        next(iterator)  # 跳过 N
    except StopIteration:
        return

    # 为了确保行结构的正确性，重新按行解析
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    rows = []
    for line in lines[1:]:
        # 兼容空格或逗号分隔
        parts = line.replace(",", " ").split()
        if parts:
            rows.append(tuple(int(x) for x in parts))

    num_rows = len(rows)
    total_score = sum(sum(r) for r in rows)

    # --- 3. 算法优化 I: 单行博弈 DP 预处理 (Endgame Database) ---
    # 如果只剩最后一行，查表 O(1) 返回，剪掉最底层的指数级搜索

    single_row_dps = []

    for cards in rows:
        L = len(cards)
        # 状态空间大小: L*(L+1)/2
        num_states = (L * (L + 1)) // 2
        dp = [0] * num_states

        # 辅助：计算三角映射索引
        # Index = Offset[l] + (r - l)
        offsets = [0] * (L + 1)
        for i in range(L):
            offsets[i + 1] = offsets[i] + (L - i)

        def get_idx(l, r):
            return offsets[l] + (r - l)

        # 长度为 1 的情况
        for i in range(L):
            dp[get_idx(i, i)] = cards[i]

        # 长度从 2 到 L
        for length in range(2, L + 1):
            for l in range(L - length + 1):
                r = l + length - 1
                # 状态转移: max(拿左, 拿右)
                val_l = cards[l] - dp[get_idx(l + 1, r)]
                val_r = cards[r] - dp[get_idx(l, r - 1)]
                dp[get_idx(l, r)] = val_l if val_l > val_r else val_r

        single_row_dps.append(dp)

    # --- 4. 预计算：直接寻址表参数 & 位运算掩码 ---
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

        # TT 大小乘数
        num_states = (L * (L + 1)) // 2

        # 预计算 Offsets 以便在 DP 查找时使用
        offsets = [0] * (L + 1)
        for k in range(L):
            offsets[k + 1] = offsets[k] + (L - k)

        # 预计算 Delta L (当 l -> l+1 时，linear_idx 的增量)
        delta_l = []
        for k in range(L):
            d = (L - k - 1) * current_mult
            delta_l.append(d)

        # Delta R (当 r -> r-1 时，linear_idx 的增量)
        delta_r = -1 * current_mult

        # 初始 local index
        init_local = L - 1
        initial_tt_idx += init_local * current_mult

        # 位状态打包 (Bit Packing): 每行占用 16 位 (High=R, Low=L)
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
                "dp": single_row_dps[i],
                "offsets": tuple(offsets),
                "mult": current_mult,
            }
        )

        current_mult *= num_states

    # 分配巨大的线性数组 (使用 array 极其节省内存且访问快)
    tt = array("i", [EMPTY_VAL]) * current_mult
    tt_set = tt.__setitem__

    ALL_CARDS = tuple(all_cards_flat)

    # 构造参数元组，用于快速解包
    # 结构: (sl, sr, dr, dl_tuple, coff, dp_list, offsets_tuple, mult)
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

    # --- 循环展开与变量绑定 (Fix: 使用哑元组消除 IDE 警告) ---
    P0 = PARAMS[0]
    # 如果没有P1, P2, P3，让它们指向P0，防止编辑器报 "unsubscriptable" 错误
    # 逻辑上 if ROW_COUNT > X 会拦截，所以这是安全的
    P1 = PARAMS[1] if ROW_COUNT > 1 else P0
    P2 = PARAMS[2] if ROW_COUNT > 2 else P0
    P3 = PARAMS[3] if ROW_COUNT > 3 else P0

    # --- 5. 核心算法: PVS (NegaScout) + 零开销状态机 ---

    def search(tt_idx, bit_state, alpha, beta):
        # 1. 极速查表
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

        # 2. 终局库检查 (Endgame Check)
        # 检查是否只剩一行
        active_cnt = 0
        last_idx = 0
        last_l, last_r = 0, 0

        # 手动展开检查，避免循环
        l0 = (bit_state >> P0[0]) & MASK_8
        r0 = (bit_state >> P0[1]) & MASK_8
        if l0 <= r0:
            active_cnt += 1
            last_idx = 0
            last_l, last_r = l0, r0

        if ROW_COUNT > 1:
            l1 = (bit_state >> P1[0]) & MASK_8
            r1 = (bit_state >> P1[1]) & MASK_8
            if l1 <= r1:
                active_cnt += 1
                last_idx = 1
                last_l, last_r = l1, r1

        if ROW_COUNT > 2:
            l2 = (bit_state >> P2[0]) & MASK_8
            r2 = (bit_state >> P2[1]) & MASK_8
            if l2 <= r2:
                active_cnt += 1
                last_idx = 2
                last_l, last_r = l2, r2

        if ROW_COUNT > 3:
            l3 = (bit_state >> P3[0]) & MASK_8
            r3 = (bit_state >> P3[1]) & MASK_8
            if l3 <= r3:
                active_cnt += 1
                last_idx = 3
                last_l, last_r = l3, r3

        # 只有一个行活跃：直接返回 DP 值
        if active_cnt == 1:
            p = PARAMS[last_idx]
            # idx = offsets[l] + (r - l)
            dp_idx = p[6][last_l] + (last_r - last_l)
            val = p[5][dp_idx]
            tt_set(tt_idx, (val << 2) | FLAG_EXACT)
            return val

        if active_cnt == 0:
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

        # 4. 移动排序 (Move Ordering)
        # 降序排列。PVS 的核心，确保大概率第一个节点就是 PV 节点
        moves.sort(reverse=True)

        # 5. PVS 逻辑
        best_val = -999999999
        original_alpha = alpha

        for i, (card, next_tt, next_bit) in enumerate(moves):
            if i == 0:
                # PV 节点：全窗口搜索
                v = card - search(next_tt, next_bit, -beta, -alpha)
            else:
                # 非 PV 节点：零窗口探测 (Null Window Search)
                # 试图证明 value < alpha (Fail Low)
                v = card - search(next_tt, next_bit, -alpha - 1, -alpha)

                # 探测失败 (Fail High)，说明此节点可能比 PV 节点好
                # 必须全窗口重搜
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

    # --- 6. 根节点逻辑 (Root Search) ---
    # 根节点虽然没有 TT 查表，但逻辑相似

    root_moves = []
    bit_state = initial_bit_state

    # 根节点生成 moves
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

    # 根节点排序
    root_moves.sort(key=lambda x: x["val"], reverse=True)

    best_diff = -float("inf")
    best_move_data = None
    alpha = -float("inf")
    beta = float("inf")

    # 根节点也应用 PVS 逻辑
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

    # --- 7. 输出结果 ---
    if best_move_data:
        my_score = (total_score + best_diff) // 2
        opp_score = total_score - my_score

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(
                f"第{best_move_data['row']}行 {best_move_data['side']} 牌点数{best_move_data['val']}\n"
            )
            f.write(f"小红: {my_score} 小蓝: {opp_score}")


if __name__ == "__main__":
    solve()
