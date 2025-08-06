import torch
import torch.nn as nn
from Myutils import visualization_tensor

class ConcatLinear(nn.Module):
    def __init__(self, single_input_dim: int = 768, output_dim: int = 768):
        """
        初始化类。

        Args:
            single_input_dim (int): 每个输入的最后一个维度大小。默认768。
            output_dim (int): 最终输出的最后一个维度大小。默认768。
        """
        super(ConcatLinear, self).__init__()
        hidden1_dim=1024
        # 第一层从 1536（拼接后的维度）降维到 1024
        self.linear1 =nn.Sequential(
            nn.Linear(single_input_dim * 2, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.LeakyReLU()
        )

        # 第二层保持 1024 到 768
        self.linear2 = nn.Sequential(
            nn.Linear(hidden1_dim, output_dim),
            nn.LayerNorm(768),
            nn.LeakyReLU()
        )
        # 第二层保持 768 到 768
        self.linear3 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU()
        )

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            input1 (torch.Tensor): 第一个输入张量，形状为 [batch_size, seq_len, input_dim]。
            input2 (torch.Tensor): 第二个输入张量，形状为 [batch_size, seq_len, input_dim]。

        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, seq_len, output_dim]。
        """
        # 沿最后一个维度拼接输入
        concatenated = torch.cat([input1, input2], dim=-1)  # [batch_size, seq_len, input_dim * 2]

        # 第一层线性变换
        visualization_tensor(input1.permute(0, 2, 1).view(-1, 16, 16),batch_dim_exit=False)
        visualization_tensor(input2.permute(0, 2, 1).view(-1, 16, 16),batch_dim_exit=False)
        hidden = self.linear1(concatenated)  # [batch_size, seq_len, hidden_dim]
        visualization_tensor(hidden.permute(0, 2, 1).view(-1, 16, 16), batch_dim_exit=False)
        hidden = self.linear2(hidden)
        visualization_tensor(hidden.permute(0, 2, 1).view(-1, 16, 16), batch_dim_exit=False)
        hidden = self.linear3(hidden)
        visualization_tensor(hidden.permute(0, 2, 1).view(-1, 16, 16), batch_dim_exit=False)
        return hidden

class ConcatLinear_0318(nn.Module):
    def __init__(self, single_input_dim: int = 768, output_dim: int = 768):
        """
        初始化类。

        Args:
            single_input_dim (int): 每个输入的最后一个维度大小。默认768。
            output_dim (int): 最终输出的最后一个维度大小。默认768。
        """
        super(ConcatLinear_0318, self).__init__()
        hidden1_dim=1024
        # 第一层从 1536（拼接后的维度）降维到 1024
        self.linear1 =nn.Sequential(
            nn.Linear(single_input_dim * 2, hidden1_dim),
            nn.LayerNorm(hidden1_dim),
            nn.LeakyReLU()
        )

        # 第二层保持 1024 到 768
        self.linear2 = nn.Sequential(
            nn.Linear(hidden1_dim, output_dim),
            nn.LayerNorm(768),
            nn.LeakyReLU()
        )
        # 第二层保持 768 到 768
        self.linear3 =  nn.Linear(output_dim, output_dim)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            input1 (torch.Tensor): 第一个输入张量，形状为 [batch_size, seq_len, input_dim]。
            input2 (torch.Tensor): 第二个输入张量，形状为 [batch_size, seq_len, input_dim]。

        Returns:
            torch.Tensor: 输出张量，形状为 [batch_size, seq_len, output_dim]。
        """
        # 沿最后一个维度拼接输入
        concatenated = torch.cat([input1, input2], dim=-1)  # [batch_size, seq_len, input_dim * 2]

        # 第一层线性变换
        # visualization_tensor(input1.permute(0, 2, 1).view(-1, 16, 16),batch_dim_exit=False)
        # visualization_tensor(input2.permute(0, 2, 1).view(-1, 16, 16),batch_dim_exit=False)
        hidden = self.linear1(concatenated)  # [batch_size, seq_len, hidden_dim]
        # visualization_tensor(hidden.permute(0, 2, 1).view(-1, 16, 16), batch_dim_exit=False)
        hidden = self.linear2(hidden)
        # visualization_tensor(hidden.permute(0, 2, 1).view(-1, 16, 16), batch_dim_exit=False)
        hidden = self.linear3(hidden)
        # visualization_tensor(hidden.permute(0, 2, 1).view(-1, 16, 16), batch_dim_exit=False)
        return hidden


class CrossModalInteraction(nn.Module):
    def __init__(self, embed_dim):
        """
        初始化跨模态交互模块
        :param embed_dim: 图像和文本特征的嵌入维度 (768)
        """
        super(CrossModalInteraction, self).__init__()
        # 图像->文本交叉注意力模块
        self.image_to_text_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        # 文本->图像交叉注意力模块
        self.text_to_image_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        # 层归一化和前馈网络
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, image_tokens, text_tokens):
        """
        前向传播
        :param image_tokens: 图像 token 特征，形状 (batch_size, num_image_tokens, embed_dim)
        :param text_tokens: 文本 token 特征，形状 (batch_size, num_text_tokens, embed_dim)
        :return: 融合后的图像和文本特征
        """
        # 图像->文本交互
        text_attended, attn_weights = self.image_to_text_attention(query=text_tokens,
                                                        key=image_tokens,
                                                        value=image_tokens)
        text_attended = self.layer_norm1(text_tokens + text_attended)  # 残差连接 + 层归一化

        # # 文本->图像交互
        # image_attended, _ = self.text_to_image_attention(query=image_tokens,
        #                                                  key=text_tokens,
        #                                                  value=text_tokens)
        # image_attended = self.layer_norm2(image_tokens + image_attended)  # 残差连接 + 层归一化

        # 融合特征通过前馈网络
        text_fused = self.feed_forward(text_attended)
        # image_fused = self.feed_forward(image_attended)

        return  text_fused

class FuseBeforeAttn(nn.Module):
    def __init__(self, embed_dim):
        """
        初始化跨模态交互模块
        :param embed_dim: 图像和文本特征的嵌入维度 (768)
        """
        super(FuseBeforeAttn, self).__init__()
        # 文本->图像交叉注意力模块
        self.text_to_image_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        # 层归一化和前馈网络
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image_tokens, text_tokens):
        """
        前向传播
        :param image_tokens: 图像 token 特征，形状 (batch_size, num_image_tokens, embed_dim)
        :param text_tokens: 文本 token 特征，形状 (batch_size, num_text_tokens, embed_dim)
        :return: 融合后的图像和文本特征
        """
        # 图像->文本交互
        b,l,c=image_tokens.shape
        # visualization_tensor(image_tokens.permute(0, 2, 1).view(b, -1, 16, 16))
        text_attended, attn_weights = self.text_to_image_attention(query=text_tokens,
                                                        key=image_tokens,
                                                        value=image_tokens)
        # for attn in attn_weights[0]:
        #     visualization_tensor(attn.view(16,16),batch_dim_exit=False)
        text_attended = text_tokens+self.layer_norm1(text_attended)  # 残差连接 + 层归一化

        # 融合特征通过前馈网络
        text_fused = self.feed_forward(text_attended)

        return  text_fused,attn_weights

class FuseBeforeAttn_V2(nn.Module):
    def __init__(self, embed_dim):
        """
        初始化跨模态交互模块
        :param embed_dim: 图像和文本特征的嵌入维度 (768)
        """
        super(FuseBeforeAttn_V2, self).__init__()
        # 文本->图像交叉注意力模块
        self.text_to_image_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        # 层归一化和前馈网络
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image_tokens, text_tokens):
        """
        前向传播
        :param image_tokens: 图像 token 特征，形状 (batch_size, num_image_tokens, embed_dim)
        :param text_tokens: 文本 token 特征，形状 (batch_size, num_text_tokens, embed_dim)
        :return: 融合后的图像和文本特征
        """
        # 图像->文本交互
        image_emb, attn_weights = self.text_to_image_attention(query=text_tokens,
                                                        key=image_tokens,
                                                        value=image_tokens)
        image_emb = self.layer_norm1(image_emb)

        # 融合特征通过前馈网络
        image_emb = image_emb+ self.feed_forward(image_emb)

        return  image_emb,attn_weights



class AttnBeforeFuse(nn.Module):
    def __init__(self, embed_dim):
        """
        初始化跨模态交互模块
        :param embed_dim: 图像和文本特征的嵌入维度 (768)
        """
        super(AttnBeforeFuse, self).__init__()
        # 文本->图像交叉注意力模块
        self.image1_to_text_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.image2_to_text_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        # 层归一化和前馈网络
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, image1_tokens, image2_tokens,text_tokens):
        """
        前向传播
        :param image_tokens: 图像 token 特征，形状 (batch_size, num_image_tokens, embed_dim)
        :param text_tokens: 文本 token 特征，形状 (batch_size, num_text_tokens, embed_dim)
        :return: 融合后的图像和文本特征
        """
        # 图像->文本交互
        attn1_feature, attn1_weights = self.image1_to_text_attn(query=text_tokens,
                                                        key=image1_tokens,
                                                        value=image1_tokens)
        attn2_feature, attn2_weights = self.image2_to_text_attn(query=text_tokens,
                                                             key=image2_tokens,
                                                             value=image2_tokens)
        image_feature=attn1_feature+attn2_feature

        # 融合特征通过前馈网络
        text_fused = self.feed_forward(image_feature)

        return  text_fused
if __name__ == '__main__':
    device="cuda"
    data=torch.randn(1,77,768).to(device)
    model=nn.MultiheadAttention(768,16,batch_first=True).to(device)
    output=model(data,data,data)
    for o in output:
        print(o.shape)
