import h5py
import numpy as np
import imageio
import os

# 设置文件名 (根据你刚才生成的日志，文件名是 rsvideowise_benchmark.h5)
FILE_NAME = "rsvideowise_benchmark.h5"
OUTPUT_GIF = "comparison_view.gif"

def create_comparison_gif():
    if not os.path.exists(FILE_NAME):
        print(f"❌ 找不到文件: {FILE_NAME}")
        return

    try:
        with h5py.File(FILE_NAME, 'r') as f:
            print(f"📂 正在读取: {FILE_NAME} ...")
            
            # 获取第一个样本的 key
            sample_keys = list(f['data'].keys())
            sample_keys.sort()
            
            if not sample_keys:
                print("❌ 数据集是空的")
                return
                
            # 选择第一个样本
            sample_name = sample_keys[0]
            print(f"🎬 正在处理样本: {sample_name}")
            
            base_path = f"data/{sample_name}"
            center_seq = f[f"{base_path}/center_sequence"][:] # 源位置
            target_seq = f[f"{base_path}/target_sequence"][:] # 目标位置
            
            # 读取属性
            attrs = f[base_path].attrs
            print(f"   -> 属性: {dict(attrs)}")
            print(f"   -> 视频形状: {center_seq.shape} (帧数, 高, 宽, 通道)")

            # --- 数据转换 (确保是 0-255 的整数图片) ---
            if center_seq.dtype != np.uint8:
                # 如果是 0-1 的浮点数，乘以 255 转为整数
                if center_seq.max() <= 1.0:
                    center_seq = (center_seq * 255).astype(np.uint8)
                    target_seq = (target_seq * 255).astype(np.uint8)
                else:
                    center_seq = center_seq.astype(np.uint8)
                    target_seq = target_seq.astype(np.uint8)

            # --- 制作并排视频 ---
            frames = []
            T, H, W, C = center_seq.shape
            
            # 创建一个白色的分割线 (宽 10 像素)
            separator = np.ones((H, 10, C), dtype=np.uint8) * 255
            
            for t in range(T):
                img_left = center_seq[t]  # 源位置第 t 帧
                img_right = target_seq[t] # 目标位置第 t 帧
                
                # 左右拼接: [左图] + [白线] + [右图]
                combined_img = np.concatenate((img_left, separator, img_right), axis=1)
                frames.append(combined_img)

            # --- 保存为 GIF ---
            print(f"💾 正在保存 GIF，请稍候...")
            imageio.mimsave(OUTPUT_GIF, frames, fps=10, loop=0)
            print(f"✅ 成功! 动图已保存为: {OUTPUT_GIF}")
            print(f"   (左边是源位置，右边是目标位置，中间是分割线)")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    create_comparison_gif()