import torch
def compare_models(model1, model2, rtol=1e-5, atol=1e-8):
    """
    比较两个模型中的同名参数是否一致，并输出结构差异。
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # 找出参数名差异
    only_in_model1 = keys1 - keys2
    only_in_model2 = keys2 - keys1
    common_keys = keys1 & keys2

    print("\n📌 仅存在于 model1 中的参数:")
    for key in sorted(only_in_model1):
        print("  -", key)

    print("\n📌 仅存在于 model2 中的参数:")
    for key in sorted(only_in_model2):
        print("  -", key)

    # 对比同名参数的值是否一致
    print("\n📌 同名参数中值不同的:")
    diff_count = 0
    for key in sorted(common_keys):
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        if not torch.allclose(param1, param2, rtol=rtol, atol=atol):
            print(f"  - 差异参数: {key}, shape: {param1.shape}")
            diff_count += 1

    if diff_count == 0:
        print("✅ 所有同名参数完全一致。")