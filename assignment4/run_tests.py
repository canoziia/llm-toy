#!/usr/bin/env python3
"""
批量运行测试案例，生成标准答案并验证
"""

import os
import sys
import time
import subprocess
from pathlib import Path


def run_test(test_id, solution_script='main.py'):
    """运行单个测试案例"""
    test_dir = f'test_cases/test{test_id:02d}'
    
    if not os.path.exists(f'{test_dir}/input.txt'):
        print(f"测试案例 {test_id} 不存在")
        return False
    
    # 复制 input.txt 到当前目录
    os.system(f'cp {test_dir}/input.txt input.txt')
    
    # 运行求解器
    start_time = time.time()
    try:
        result = subprocess.run(
            ['python3', solution_script],
            timeout=10,  # 10秒超时
            capture_output=True,
            text=True
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"测试 {test_id:02d}: 运行失败")
            print(result.stderr)
            return False, 0, elapsed_time
        
        # 复制 output.txt 到测试目录作为标准答案
        if os.path.exists('output.txt'):
            os.system(f'cp output.txt {test_dir}/output.txt')
            print(f"测试 {test_id:02d}: 完成 (耗时: {elapsed_time:.2f}s)")
            print(f"  {result.stdout.strip()}")
            return True, elapsed_time
        else:
            print(f"测试 {test_id:02d}: 未生成 output.txt")
            return False, elapsed_time
            
    except subprocess.TimeoutExpired:
        print(f"测试 {test_id:02d}: 超时 (>10s)")
        return False, 10.0
    except Exception as e:
        print(f"测试 {test_id:02d}: 异常 - {e}")
        return False, 0.0


def main():
    """批量运行所有测试"""
    print("=" * 60)
    print("开始生成标准答案")
    print("=" * 60)
    
    total_tests = 20
    passed = 0
    failed = 0
    stats = []  # 存储统计信息: (test_id, time)
    
    for test_id in range(1, total_tests + 1):
        success, elapsed_time = run_test(test_id)
        if success:
            passed += 1
            stats.append((test_id, elapsed_time))
        else:
            failed += 1
        print()
    
    # 清理临时文件
    if os.path.exists('input.txt'):
        os.remove('input.txt')
    if os.path.exists('output.txt'):
        os.remove('output.txt')
    
    print("=" * 60)
    print(f"总计: {total_tests} 个测试")
    print(f"成功: {passed} 个")
    print(f"失败: {failed} 个")
    print("=" * 60)
    
    if stats:
        print("\n性能统计分析:")
        print("-" * 60)
        print(f"{'测试ID':<10} {'时间(s)':<10}")
        print("-" * 60)
        for test_id, elapsed_time in stats:
            print(f"{test_id:<10} {elapsed_time:<10.2f}")
        print("-" * 60)
        
        # 计算总体统计
        total_time = sum(s[1] for s in stats)
        print(f"总时间: {total_time:.2f}s")
        print("=" * 60)


if __name__ == '__main__':
    main()
