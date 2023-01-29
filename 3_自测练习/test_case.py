#!/usr/bin/env python3

from my_solution import test
import torch
import sys


# 测试用例:需要使用绝对路径，相对路径会报错
def test_solution():
    res = test()
    print(res)
    assert res == [['浙商银行', '叶老桂', '叶老桂']]  # 判断输出结果，预期increment(8)应该为9


if __name__ == '__main__':
    test_solution()